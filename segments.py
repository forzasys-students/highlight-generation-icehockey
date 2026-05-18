import cv2
import torch
from collections import defaultdict

from sportsbd import run_video_inference

from camera_model import get_video_duration_ms, sample_frames_for_shot_with_times, classify_frames_batch, majority_label
from ffmpeg_utils import ffprobe_duration_ms
from utils import boundary_times_ms, logo_times_ms


# Labels grouped into higher-level recap categories.
MAIN_LABELS = {"behind_the_goal", "main_camera_center", "main_camera_left", "main_camera_right"}
CLOSE_LABELS = {"close_up_player_or_field_referee", "close_up_side_or_staff"}
PUBLIC_LABELS = {"public_or_fans"}


def shot_label_to_cat(lbl):
    """
    Map fine-grained classifier labels to broader recap categories.
    """
    if lbl in MAIN_LABELS:
        return "main"
    if lbl in CLOSE_LABELS:
        return "close"
    if lbl in PUBLIC_LABELS:
        return "public"
    return "other"



def build_shots_from_boundaries(boundaries, start_ms=0, end_ms=None):
    """
    Convert boundary timestamps into contiguous shot intervals.

    Example:
    boundaries = [1000, 2500], start=0, end=4000
    -> [(0,1000), (1000,2500), (2500,4000)]

    Notes:
    - boundaries outside [start_ms, end_ms] are ignored
    - zero/negative-length intervals are skipped
    """
    if end_ms is None:
        raise ValueError("end_ms must be provided")

    valid = sorted(set(int(t) for t in boundaries if start_ms < int(t) < end_ms))

    shots = []
    prev = int(start_ms)

    for t in valid:
        if t > prev:
            shots.append((prev, t))
        prev = t

    if end_ms > prev:
        shots.append((prev, int(end_ms)))

    return shots

def detect_mixed_shot_split(sample_times_ms, sample_labels):
    """
    Detect whether a shot should be split into two parts.

    Heuristic:
    - Convert labels to broad categories (main / close / public / other)
    - If the beginning and the end are consistently different,
      return a split timestamp.
    - Otherwise return None.
    """
    if len(sample_times_ms) < 4 or len(sample_times_ms) != len(sample_labels):
        return None

    cats = [shot_label_to_cat(lbl) for lbl in sample_labels]

    # No point splitting if all sampled categories are the same.
    uniq = set(cats)
    if len(uniq) <= 1:
        return None

    start_cat = cats[0]
    end_cat = cats[-1]

    # If the start and end look the same, do not split.
    if start_cat == end_cat:
        return None

    # Stronger evidence:
    # first two samples agree, last two samples agree, and they differ.
    if len(cats) >= 4:
        if cats[0] == cats[1] and cats[-1] == cats[-2] and cats[0] != cats[-1]:
            last_start_idx = max(i for i, c in enumerate(cats) if c == start_cat)
            later_end_candidates = [i for i, c in enumerate(cats) if c == end_cat and i > last_start_idx]
            if not later_end_candidates:
                return None

            first_end_idx = min(later_end_candidates)

            t1 = sample_times_ms[last_start_idx]
            t2 = sample_times_ms[first_end_idx]
            split_t = (t1 + t2) // 2
            return split_t

    return None


@torch.no_grad()
def refine_and_classify_shots(
    cap,
    model,
    device,
    preprocess,
    LABELS,
    shots,
    batch_size=64,
    num_samples=7,
):
    """
    Classify shots and split only when a shot looks mixed.

    Returns:
    - refined_shots: list of (start_ms, end_ms)
    - refined_labels: one label per refined shot
    """
    sampled_frames = []
    sampled_meta = []  # (shot_idx, timestamp_ms)

    for shot_idx, (a, b) in enumerate(shots):
        samples = sample_frames_for_shot_with_times(cap, a, b, num_samples=num_samples)
        for t_ms, fr in samples:
            sampled_frames.append(fr)
            sampled_meta.append((shot_idx, t_ms))

    if not sampled_frames:
        return shots, ["main_camera_center"] * len(shots)

    preds_all = []
    start = 0
    while start < len(sampled_frames):
        chunk = sampled_frames[start:start + batch_size]
        preds = classify_frames_batch(model, chunk, device, preprocess, LABELS)
        preds_all.extend(preds)
        start += batch_size

    by_shot = defaultdict(list)
    for (shot_idx, t_ms), (lbl, conf) in zip(sampled_meta, preds_all):
        by_shot[shot_idx].append((t_ms, lbl, conf))

    refined_shots = []
    refined_labels = []

    for shot_idx, (a, b) in enumerate(shots):
        items = sorted(by_shot.get(shot_idx, []), key=lambda x: x[0])

        if not items:
            refined_shots.append((a, b))
            refined_labels.append("main_camera_center")
            continue

        sample_times = [t for t, _, _ in items]
        sample_labels = [lbl for _, lbl, _ in items]

        split_t = detect_mixed_shot_split(sample_times, sample_labels)

        if split_t is None or split_t <= a or split_t >= b:
            # Keep original shot unchanged.
            lbl = majority_label(sample_labels, default="main_camera_center")
            refined_shots.append((a, b))
            refined_labels.append(lbl)
        else:
            # Split into left and right parts and classify each half separately.
            left_labels = [lbl for t, lbl, _ in items if t < split_t]
            right_labels = [lbl for t, lbl, _ in items if t >= split_t]

            left_lbl = majority_label(left_labels, default="main_camera_center")
            right_lbl = majority_label(right_labels, default="main_camera_center")

            if split_t - a > 0:
                refined_shots.append((a, split_t))
                refined_labels.append(left_lbl)

            if b - split_t > 0:
                refined_shots.append((split_t, b))
                refined_labels.append(right_lbl)

    return refined_shots, refined_labels

def find_core_main_run(
    shots,
    shot_labels,
    core_start,
    core_end,
    min_valid_core_main_ms=6000,
    tolerate_nonmain_ms=1200,
):
    """
    Find the first stable main-camera run inside the core window.

    Strategy:
    - Start looking for a run when the first "main" shot appears.
    - Allow short non-main interruptions inside the run.
    - If too much non-main accumulates, close the run.
    - Return the first run that is long enough to be considered the real core_main.

    Returns:
    - (start_ms, end_ms) if found
    - (core_start, core_end) as fallback
    """
    run_start = None
    run_end = None
    nonmain_inside_run = 0

    for (a, b), lbl in zip(shots, shot_labels):
        a = int(a)
        b = int(b)

        if a >= core_end:
            break
        if b <= a:
            continue
        if b <= core_start:
            continue

        a = max(a, core_start)
        b = min(b, core_end)
        if b <= a:
            continue

        cat = shot_label_to_cat(lbl)
        dur = b - a

        if run_start is None:
            # Start candidate run only when the first main shot appears.
            if cat == "main":
                run_start = a
                run_end = b
                nonmain_inside_run = 0
            continue

        # Continue candidate run.
        if cat == "main":
            run_end = b
        else:
            nonmain_inside_run += dur

            # Allow short interruptions, but stop the candidate if too much
            # non-main content accumulates.
            if nonmain_inside_run > tolerate_nonmain_ms:
                run_len = (run_end - run_start) if (run_start is not None and run_end is not None) else 0
                if run_len >= min_valid_core_main_ms:
                    return run_start, run_end

                # Reset and keep searching for a better run.
                run_start = None
                run_end = None
                nonmain_inside_run = 0

    # Final candidate at end of loop.
    if run_start is not None and run_end is not None:
        run_len = run_end - run_start
        if run_len >= min_valid_core_main_ms:
            return run_start, run_end

    # Fallback if no good run was found.
    return core_start, core_end


def extract_two_logo_segments(
    clip_path,
    sbd_checkpoint_path,
    model, device, preprocess, LABELS,
    sbd_threshold,
    min_gap_ms,
    min_segment_ms,
    pad_ms,
    edge_guard_ms=80,
    core_main_nonmain_ms=1500,
    min_valid_core_main_ms=6000,
    tolerate_nonmain_ms=1200,
    early_logo_cutoff_ms=10000,
):
    """
    Extract categorized recap segments from a goal clip.

    Logic:
    - Detect boundaries and logo transitions.
    - If a logo appears very early in the clip, treat it as a bad trim marker
      and move the effective clip start to that logo instead of treating it as replay start.
    - Define:
        * core window   = from effective start until first valid replay logo
        * replay window = between first and second valid replay logos
    - Classify shots.
    - Find the first stable "real" core_main run, even if the clip starts earlier.
    - Ignore early main shots before core_main_start.
    - Split later main shots into core_after.
    """
    detections = run_video_inference(
        video_path=clip_path,
        checkpoint_path=sbd_checkpoint_path,
        threshold=sbd_threshold,
        stride=4,
        t_frames=16,
        fps=25,
    )

    boundaries = boundary_times_ms(detections)

    filtered = []
    for t in boundaries:
        if not filtered or (t - filtered[-1]) >= min_gap_ms:
            filtered.append(t)
    boundaries = filtered

    all_logos = sorted(int(t) for t in logo_times_ms(detections) if t is not None)

    # Logos very early in the clip are treated as "bad trim" markers.
    early_logos = [t for t in all_logos if t < int(early_logo_cutoff_ms)]
    valid_replay_logos = [t for t in all_logos if t >= int(early_logo_cutoff_ms)]

    # Move the effective start of the clip to the last early logo if present.
    # This trims away junk before the real beginning of the play.
    effective_start_ms = early_logos[-1] + int(pad_ms) if early_logos else 0

    print(
        f"[logos] all={[round(t/1000, 2) for t in all_logos]} "
        f"early={[round(t/1000, 2) for t in early_logos]} "
        f"valid_replay={[round(t/1000, 2) for t in valid_replay_logos]} "
        f"effective_start={effective_start_ms/1000:.2f}s"
    )

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {clip_path}")

    duration_ms = get_video_duration_ms(cap)
    if duration_ms <= 0:
        cap.release()
        duration_ms = ffprobe_duration_ms(clip_path)
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {clip_path}")

    logo1 = valid_replay_logos[0] if len(valid_replay_logos) >= 1 else None
    logo2 = valid_replay_logos[1] if len(valid_replay_logos) >= 2 else None

    core_end = duration_ms
    replay_start = None
    replay_end = None
    hard_cut_end = duration_ms

    if logo1 is not None:
        core_end = max(effective_start_ms, int(logo1) - int(pad_ms))
        replay_start = min(duration_ms, int(logo1) + int(pad_ms))

    if logo2 is not None:
        replay_end = max(0, int(logo2) - int(pad_ms))
        hard_cut_end = max(0, int(logo2) - int(pad_ms))
    else:
        hard_cut_end = duration_ms
        replay_end = duration_ms if replay_start is not None else None

    core_win = (effective_start_ms, core_end)
    replay_win = None
    if replay_start is not None and replay_end is not None and replay_end > replay_start:
        replay_win = (replay_start, replay_end)

    shots = build_shots_from_boundaries(boundaries, start_ms=effective_start_ms, end_ms=hard_cut_end)

    shots, shot_labels = refine_and_classify_shots(
        cap, model, device, preprocess, LABELS, shots, batch_size=64, num_samples=7
    )
    cap.release()

    # Find the real core_main run, even if the clip starts with unrelated content.
    core_main_start, initial_core_main_end = find_core_main_run(
        shots,
        shot_labels,
        core_start=core_win[0],
        core_end=core_win[1],
        min_valid_core_main_ms=int(min_valid_core_main_ms),
        tolerate_nonmain_ms=int(tolerate_nonmain_ms),
    )

    # Extend core_main until a long enough continuous non-main run appears.
    core_main_end = initial_core_main_end
    non_main_run_ms = 0
    inside_core_main_zone = False

    for (a0, b0), lbl in zip(shots, shot_labels):
        a0 = int(a0)
        b0 = int(b0)

        if a0 >= core_win[1]:
            break
        if b0 <= a0:
            continue

        a1 = max(a0, core_main_start)
        b1 = min(b0, core_win[1])
        if b1 <= a1:
            continue

        # Ignore everything before the detected start of the real core_main.
        if b0 <= core_main_start:
            continue

        cat0 = shot_label_to_cat(lbl)
        dur0 = b1 - a1

        if not inside_core_main_zone:
            inside_core_main_zone = True

        if cat0 == "main":
            core_main_end = b1
            non_main_run_ms = 0
        else:
            non_main_run_ms += dur0
            if non_main_run_ms >= core_main_nonmain_ms:
                core_main_end = a1
                break

    # Safety clamp.
    core_main_start = max(core_win[0], min(core_main_start, core_win[1]))
    core_main_end = max(core_main_start, min(core_main_end, core_win[1]))

    print(
        f"[core-main] start={core_main_start/1000:.2f}s "
        f"end={core_main_end/1000:.2f}s "
        f"core_start={core_win[0]/1000:.2f}s "
        f"core_end={core_win[1]/1000:.2f}s "
        f"min_valid={min_valid_core_main_ms}ms "
        f"tolerate_nonmain={tolerate_nonmain_ms}ms "
        f"stop_nonmain={core_main_nonmain_ms}ms"
    )

    all_segments = defaultdict(list)
    guard = int(edge_guard_ms)

    for (a0, b0), lbl in zip(shots, shot_labels):
        a0 = int(a0)
        b0 = int(b0)
        if b0 <= a0:
            continue

        a = a0 + guard
        b = b0 - guard
        if b <= a:
            continue

        dur = b - a
        if dur < min_segment_ms:
            continue

        cat = shot_label_to_cat(lbl)
        if cat == "other":
            continue

        # -------------------------
        # CORE WINDOW
        # -------------------------
        if b <= core_win[1]:
            # Drop everything before the detected start of the real core_main.
            if b <= core_main_start:
                continue

            if cat == "main":
                # Fully inside real core_main.
                if a >= core_main_start and b <= core_main_end:
                    all_segments[("main", "core")].append({
                        "a": a, "b": b, "dur": dur, "type": "main", "region": "core"
                    })
                    continue

                # Fully after real core_main.
                if a >= core_main_end:
                    all_segments[("main", "core_after")].append({
                        "a": a, "b": b, "dur": dur, "type": "main", "region": "core_after"
                    })
                    continue

                # Crosses into core_main start.
                if a < core_main_start < b <= core_main_end:
                    aa = core_main_start
                    bb = b
                    if (bb - aa) >= min_segment_ms:
                        all_segments[("main", "core")].append({
                            "a": aa, "b": bb, "dur": (bb - aa), "type": "main", "region": "core"
                        })
                    continue

                # Crosses out of core_main end.
                if core_main_start <= a < core_main_end < b:
                    aa1, bb1 = a, core_main_end
                    if (bb1 - aa1) >= min_segment_ms:
                        all_segments[("main", "core")].append({
                            "a": aa1, "b": bb1, "dur": (bb1 - aa1), "type": "main", "region": "core"
                        })

                    aa2, bb2 = core_main_end, b
                    if (bb2 - aa2) >= min_segment_ms:
                        all_segments[("main", "core_after")].append({
                            "a": aa2, "b": bb2, "dur": (bb2 - aa2), "type": "main", "region": "core_after"
                        })
                    continue

                # Covers both start and end.
                if a < core_main_start and b > core_main_end:
                    aa1, bb1 = core_main_start, core_main_end
                    if (bb1 - aa1) >= min_segment_ms:
                        all_segments[("main", "core")].append({
                            "a": aa1, "b": bb1, "dur": (bb1 - aa1), "type": "main", "region": "core"
                        })

                    aa2, bb2 = core_main_end, b
                    if (bb2 - aa2) >= min_segment_ms:
                        all_segments[("main", "core_after")].append({
                            "a": aa2, "b": bb2, "dur": (bb2 - aa2), "type": "main", "region": "core_after"
                        })
                    continue

            else:
                # Non-main content before core_main_start should be dropped too.
                aa = max(a, core_main_start)
                bb = b

                if (bb - aa) >= min_segment_ms:
                    all_segments[(cat, "core")].append({
                        "a": aa, "b": bb, "dur": (bb - aa), "type": cat, "region": "core"
                    })
                continue

        # -------------------------
        # CROSSING CORE -> REPLAY
        # -------------------------
        if a < core_win[1] < b:
            if cat == "main":
                # Main inside core part only counts if it overlaps the real core_main range.
                aa_core = max(a, core_main_start)
                bb_core = min(b, core_main_end, core_win[1])
                if (bb_core - aa_core) >= min_segment_ms:
                    all_segments[("main", "core")].append({
                        "a": aa_core, "b": bb_core, "dur": (bb_core - aa_core), "type": "main", "region": "core"
                    })

                aa_after = max(a, core_main_end)
                bb_after = min(b, core_win[1])
                if (bb_after - aa_after) >= min_segment_ms:
                    all_segments[("main", "core_after")].append({
                        "a": aa_after, "b": bb_after, "dur": (bb_after - aa_after), "type": "main", "region": "core_after"
                    })
            else:
                a1, b1 = a, core_win[1]
                if (b1 - a1) >= min_segment_ms:
                    all_segments[(cat, "core")].append({
                        "a": a1, "b": b1, "dur": (b1 - a1), "type": cat, "region": "core"
                    })

            if replay_win is not None:
                aa = max(core_win[1], replay_win[0])
                bb = min(b, replay_win[1])
                if (bb - aa) >= min_segment_ms:
                    all_segments[(cat, "replay")].append({
                        "a": aa, "b": bb, "dur": (bb - aa), "type": cat, "region": "replay"
                    })
            continue

        # -------------------------
        # REPLAY WINDOW
        # -------------------------
        if replay_win is not None:
            aa = max(a, replay_win[0])
            bb = min(b, replay_win[1])
            if (bb - aa) >= min_segment_ms:
                all_segments[(cat, "replay")].append({
                    "a": aa, "b": bb, "dur": (bb - aa), "type": cat, "region": "replay"
                })

    meta = {
        "duration_ms": duration_ms,
        "effective_start_ms": effective_start_ms,
        "logo1": logo1,
        "logo2": logo2,
        "all_logos": all_logos,
        "early_logos": early_logos,
        "core_end": core_end,
        "core_main_start": core_main_start,
        "core_main_end": core_main_end,
        "replay_start": replay_start,
        "replay_end": replay_end,
        "hard_cut_end": hard_cut_end,
    }
    return meta, all_segments