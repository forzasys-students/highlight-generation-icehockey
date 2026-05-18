import os
import shutil

import torch
from sportsbd import download_model
from camera_model import build_preprocess, try_load_model
from events import load_events, parse_events, select_all_goals, enrich_goal_scores, find_team_info
from ffmpeg_utils import (
    ffmpeg_download,
    concat_with_copy_fallback,
    make_intro,
    render_goal_clip,
    ffprobe_duration_ms,
    prepare_logo_png,
    xfade_sequence,
    ffmpeg_concat_reencode,
    ffprobe_video_fps
)
from segments import extract_two_logo_segments, find_core_main_run
from utils import (
    ensure_dir,
    export_segments_xlsx,
    segs_sorted,
    segs_to_intervals,
    intervals_len,
)


def run(args):
    # --- Load event data, keep goal events, and derive score metadata for each goal. ---
    data = load_events(args.events_url or args.events_json)
    events = parse_events(data)
    goals = select_all_goals(events)
    if not goals:
        raise RuntimeError("No goals found (action='goal').")

    goals = enrich_goal_scores(goals)

    highlight_sec = args.highlight_sec if args.highlight_sec is not None else len(goals) * 22.0

    for i, g in enumerate(goals, start=1):
        print(
            f"[score] goal#{i} raw={g.get('score')} "
            f"before={g.get('score_before_display')} "
            f"after={g.get('score_after_display')} "
            f"scorer={g.get('scorer')}"
        )
    
    # --- Set up the working directory, select the inference device, and load the camera classification model and sportsbd checkpoint. ---
    ensure_dir(args.workdir)

    LABELS = [
        "behind_the_goal",
        "close_up_player_or_field_referee",
        "close_up_side_or_staff",
        "main_camera_center",
        "main_camera_left",
        "main_camera_right",
        "public_or_fans",
    ]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("[device]", device)

    preprocess = build_preprocess()
    
    model = try_load_model(args.model, device, num_classes=len(LABELS))
    sbd_checkpoint_path = download_model()
    print(f"[sportsbd] using checkpoint {sbd_checkpoint_path} for shot boundary detection")

    # --- Convert time values to milliseconds and derive the available highlight budget. ---
    min_seg_ms = int(args.min_segment_sec * 1000.0)
    intro_time = 3.0
    target_total_ms = int(highlight_sec * 1000.0) - int(intro_time * 1000.0)
    min_valid_core_main_ms = int(args.min_valid_core_main_sec * 1000.0)
    tolerate_nonmain_ms = int(args.tolerate_nonmain_sec * 1000.0)

    # --- Log the goal count, highlight budget, and transition settings. ---
    print("[goals]", len(goals))
    print("[budget] time budget:", f"{target_total_ms/1000:.2f}s (reduced by 3s to fit logo) (CORE MAIN can exceed) ")

    if args.transition_type == "none":
        print("[transition] disabled")
    else:
        print(f"[transition] type={args.transition_type} sec={args.transition_sec}")

    # --- Extract team information and set up the logo and overlay assets used in the final video. ---
    team = find_team_info(data)
    home_name = team["home_name"]
    away_name = team["away_name"]
    home_short = team["home_short"]
    away_short = team["away_short"]
    match_name = team["match_name"]

    assets_dir = os.path.join(args.workdir, "_assets")
    ensure_dir(assets_dir)

    home_logo_path = os.path.join(assets_dir, "home_logo.png")
    away_logo_path = os.path.join(assets_dir, "away_logo.png")

    prepare_logo_png(team["home_logo"], home_logo_path, home_name)
    prepare_logo_png(team["away_logo"], away_logo_path, away_name)
    BACKGROUND_IMG = os.path.join("assets", "background.png")
    SCOREBOARD_IMG = os.path.join("assets", "scoreboard.png")
    SCOREDBY_IMG = os.path.join("assets", "scoredBy.png")

    # --- Download the source clip for each goal event and collect the local file paths. ---
    raw_paths = []
    for i, ev in enumerate(goals, start=1):
        base = f"{i:03d}_GOAL_{ev['phase_ord']}_{ev['game_time']}"
        raw_path = os.path.join(args.workdir, base + ".mp4")
        print(f"[dl] {i}/{len(goals)} -> {raw_path}")
        ffmpeg_download(ev["url"], raw_path)
        raw_paths.append(raw_path)

    # --- Read the input frame rate from the first downloaded clip. ---
    input_fps = ffprobe_video_fps(raw_paths[0])
    if input_fps <= 0:
        raise RuntimeError(f"Could not read fps from {raw_paths[0]}")
    
    # --- Analyze each goal clip and extract the segment metadata used for clip selection. ---
    meta_by_goal = {}
    all_segments_by_goal = {}
    for gi, path in enumerate(raw_paths):
        meta, all_segments = extract_two_logo_segments(
            clip_path=path,
            sbd_checkpoint_path = sbd_checkpoint_path,
            model=model,
            device=device,
            preprocess=preprocess,
            LABELS=LABELS,
            sbd_threshold=args.sbd_threshold,
            min_gap_ms=args.min_gap_ms,
            min_segment_ms=min_seg_ms,
            pad_ms=int(args.pad_ms),
            edge_guard_ms=int(args.edge_guard_ms),
            core_main_nonmain_ms=int(args.core_main_nonmain_ms),
            min_valid_core_main_ms=min_valid_core_main_ms,
            tolerate_nonmain_ms=tolerate_nonmain_ms,
        )

        # Fall back to an early main segment if no valid main/core segment was detected.
        if not all_segments.get(("main", "core")):
            core_end = max(2000, min(meta["core_end"], 8000))
            all_segments[("main", "core")] = [{
                "a": 0, "b": core_end, "dur": core_end, "type": "main", "region": "core"
            }]
            meta["core_main_end"] = core_end

        meta_by_goal[gi] = meta
        all_segments_by_goal[gi] = all_segments

        l1 = meta["logo1"]
        l2 = meta["logo2"]
        print(
            f"[logo] goal#{gi+1} logo1={None if l1 is None else round(l1/1000, 2)}s "
            f"logo2={None if l2 is None else round(l2/1000, 2)}s "
            f"core_end={meta['core_end']/1000:.2f}s "
            f"core_main_start={meta['core_main_start']/1000:.2f}s "
            f"core_main_end={meta['core_main_end']/1000:.2f}s "
            f"replay=({None if meta['replay_start'] is None else round(meta['replay_start']/1000, 2)}s"
            f"->{None if meta['replay_end'] is None else round(meta['replay_end']/1000, 2)}s) "
            f"hard_cut_end={meta['hard_cut_end']/1000:.2f}s"
        )

    # --- Optionally export the segment analysis results for all goal clips to an XLSX file. ---
    if args.export_xlsx:
        xlsx_path = os.path.join(args.workdir, "segments.xlsx")
        export_segments_xlsx(xlsx_path, goals, raw_paths, meta_by_goal, all_segments_by_goal)
        print("[xlsx] wrote", xlsx_path)
   
    # --- Define the segment priority order and select the main/core intervals for each goal clip first. ---
    # These priority lists control which extra segments are added before replay material.
    CORE_EXTRA_ORDER = [("close", "core"), ("public", "core"), ("main", "core_after")]
    REPLAY_ORDER = [("main", "replay"), ("close", "replay"), ("public", "replay")]

    selected_intervals_by_goal = {}
    base_total = 0

    for gi in range(len(raw_paths)):
        hard_end = meta_by_goal[gi]["hard_cut_end"]
        main_core = segs_sorted(all_segments_by_goal[gi].get(("main", "core"), []))
        base_total += sum(int(s["dur"]) for s in main_core)
        selected_intervals_by_goal[gi] = segs_to_intervals(main_core, hard_end)

    print(f"[base] CORE MAIN total = {base_total/1000:.2f}s | budget(extras) = {target_total_ms/1000:.2f}s")

    # --- Fill the remaining time budget with extra core segments before considering replay footage. --- 
    total_used = base_total
    remaining = max(0, target_total_ms - total_used)
    if remaining < 5 * 1000.0:
        print(f"[base] Remaining time left = {remaining/1000:.2f}s | set to 5 seconds to fit an extra clip at the end to update the score")
        remaining = 5 * 1000.0
    if remaining > 0:
        print(f"[extras-core] remaining = {remaining/1000:.2f}s")
        extra_queues = {}
        extra_pos = {}

        # Gather the extra segment candidates for each goal clip in priority order. 
        for gi in range(len(raw_paths)):
            hard_end = meta_by_goal[gi]["hard_cut_end"]
            for key in CORE_EXTRA_ORDER:
                seq = []
                for s in segs_sorted(all_segments_by_goal[gi].get(key, [])):
                    a = max(0, int(s["a"]))
                    b = min(int(hard_end), int(s["b"]))
                    if b > a:
                        seq.append([a, b])
                extra_queues[(gi, key)] = seq
                extra_pos[(gi, key)] = 0

        # Keep adding the next segment that fits until the remaining budget is used up.
        progress = True
        while remaining > 0 and progress:
            progress = False

            for gi in range(len(raw_paths) - 1, -1, -1):
                if remaining <= 0:
                    break

                cur = list(selected_intervals_by_goal[gi])

                for key in CORE_EXTRA_ORDER:
                    idx = extra_pos[(gi, key)]
                    seq = extra_queues[(gi, key)]
                    if idx >= len(seq):
                        continue

                    a, b = seq[idx]
                    dur = b - a
                    if dur <= remaining:
                        cur.append([a, b])
                        selected_intervals_by_goal[gi] = cur
                        extra_pos[(gi, key)] += 1
                        remaining -= dur
                        total_used += dur
                        progress = True
                        break

    # --- Add replay segments only if the full replay for a goal clip fits within the remaining budget. ---
    remaining = max(0, target_total_ms - total_used)
    if remaining > 0:
        print(f"[extras-replay] remaining = {remaining/1000:.2f}s (replay all-or-nothing)")
        for gi in range(len(raw_paths)):
            if remaining <= 0:
                break
            hard_end = meta_by_goal[gi]["hard_cut_end"]

            # Collect the full replay block for the current goal clip before checking if it fits.
            replay_intervals = []
            replay_total = 0
            for key in REPLAY_ORDER:
                for s in segs_sorted(all_segments_by_goal[gi].get(key, [])):
                    a = max(0, int(s["a"]))
                    b = min(int(hard_end), int(s["b"]))
                    if b > a:
                        replay_intervals.append([a, b])
                        replay_total += (b - a)

            if replay_total > 0 and replay_total <= remaining:
                selected_intervals_by_goal[gi].extend(replay_intervals)
                remaining -= replay_total
                total_used += replay_total
    
    # --- Calculate and print how much footage was selected for each goal clip and in total. ---
    total_planned = 0
    for gi in range(len(raw_paths)):
        d = intervals_len(selected_intervals_by_goal[gi])
        total_planned += d
        print(f"[plan] goal#{gi+1} selected={d/1000:.2f}s")

    print(f"[plan] TOTAL selected = {total_planned/1000:.2f}s | budget={target_total_ms/1000:.2f}s | base={base_total/1000:.2f}s")

    # --- Measure the main/core length of each goal clip to decide when the on-screen score changes and the scorer text appears. ---
    selected_main_core_len_by_goal = {}
    for gi in range(len(raw_paths)):
        main_core_intervals = segs_to_intervals(
            all_segments_by_goal[gi].get(("main", "core"), []),
            meta_by_goal[gi]["hard_cut_end"]
        )
        selected_main_core_len_by_goal[gi] = intervals_len(main_core_intervals)

    # --- Render each goal clip with the selected segments, score graphics, and scorer information. ---
    rendered_goal_clips = []
    print(f"[plan] Rendering goal clips with logos and scorer/score lines")
    for gi, raw_path in enumerate(raw_paths):
        intervals = selected_intervals_by_goal[gi]

        # Prepare scores, scorer text, and when the on-screen information should change.
        scorer = goals[gi].get("scorer") or "UNKNOWN SCORER"
        score_before = goals[gi].get("score_before_display") or "0-0"
        score_after = goals[gi].get("score_after_display") or score_before

        scorer_line = f"Scored by {scorer}"
        switch_sec = selected_main_core_len_by_goal[gi] / 1000.0

        tmp_final = os.path.splitext(raw_path)[0] + ".final.mp4"
        render_goal_clip(
            raw_path,
            intervals,
            tmp_final,
            home_logo_path,
            away_logo_path,
            home_short,
            away_short,
            scorer_line,
            score_before,
            score_after,
            switch_sec,
            scoreboard_bg=SCOREBOARD_IMG,
            scorer_bg=SCOREDBY_IMG,
            scorer_transition_sec=0.4,
        )

        rendered_goal_clips.append(tmp_final)
    
    # --- Create the intro clip with the custom background graphic, team logos, and match information. ---
    intro_path = os.path.join(args.workdir, "_intro.mp4")
    print(f"[plan] Creating intro...")
    make_intro(
        out_path=intro_path, 
        background_img=BACKGROUND_IMG, 
        home_logo=home_logo_path, 
        away_logo=away_logo_path, 
        home_name=home_name, 
        away_name=away_name, 
        match_name=match_name, 
        fps = input_fps, 
        dur_sec=intro_time,
    )

    # --- Combine the intro and goal clips into the final highlight video and log its length. ---
    print(f"[plan] Concatenating final video with transitions")
    all_clips = [intro_path] + rendered_goal_clips

    if args.transition_type == "none":
        ffmpeg_concat_reencode(all_clips, args.out)
    else:
        xfade_sequence(all_clips, args.out, transition_sec = args.transition_sec, transition_type = args.transition_type)

    real = ffprobe_duration_ms(args.out)
    print(f"[done] wrote {args.out} real={real/1000:.2f}s (budget={target_total_ms/1000:.2f}s; base={base_total/1000:.2f}s)")

    # --- Delete the temporary workdir unless --keep_workdir is set. ---
    if not args.keep_workdir:
        shutil.rmtree(args.workdir, ignore_errors=True)
