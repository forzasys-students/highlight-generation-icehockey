import json
import os
import shutil
import subprocess
from urllib.request import Request

from urllib.parse import urljoin

from utils import ensure_dir, escape_drawtext, get_drawtext_font_arg, download_image


# =========================================================
# Encoding / quality defaults
# =========================================================
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "slow"
VIDEO_CRF = "10"  # 8-12 = very good quality, larger files

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
AUDIO_RATE = "48000"

# =========================================================
# DEBUG / logging
# =========================================================

DEBUG = False  # CHANGE TO True if you need more logs

FFMPEG_LOGLEVEL = "warning" if DEBUG else "fatal"
FFPROBE_LOGLEVEL = "warning" if DEBUG else "error"

def run(cmd):
    return run_cmd(cmd, check=True)

FFMPEG_BASE = [
    "ffmpeg",
    "-y",
    "-hide_banner",
    "-loglevel", FFMPEG_LOGLEVEL,
    "-nostats",
]


def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def run_cmd(cmd, check=True):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if DEBUG:
        if p.stdout:
            print(p.stdout)
        if p.stderr:
            print(p.stderr)
    if check and p.returncode != 0:
        raise RuntimeError((p.stderr or "").strip() or f"Command failed with exit code {p.returncode}")
    return p


# =========================================================
# Image helpers
# =========================================================

def make_text_placeholder_png(out_path, text):
    """
    Create a simple placeholder PNG with centered text using ffmpeg.
    Used when a team logo cannot be downloaded/decoded.
    """
    fontarg = get_drawtext_font_arg()
    safe = escape_drawtext(text)
    vf = (
        f"drawtext={fontarg}text='{safe}':"
        f"x=(w-text_w)/2:y=(h-text_h)/2:fontsize=64:"
        f"fontcolor=black:box=1:boxcolor=white@1.0:boxborderw=18"
    )
    run(FFMPEG_BASE + [
        "-f", "lavfi", "-i", "color=c=gray:s=256x256",
        "-vf", vf,
        "-frames:v", "1",
        "-vcodec", "png",
        out_path
    ])


def convert_any_image_to_png(in_path, out_path):
    """
    Convert an arbitrary image format into a PNG file using ffmpeg.
    """
    run(FFMPEG_BASE + [
        "-i", in_path,
        "-frames:v", "1",
        "-vcodec", "png",
        out_path
    ])


def prepare_logo_png(logo_url, out_png_path, fallback_text):
    """
    Prepare a team logo PNG:
    - try to download original logo
    - convert to PNG
    - if anything fails, create a text placeholder
    """
    tmp_path = out_png_path + ".tmp"
    try:
        if not logo_url:
            raise RuntimeError("logo_url missing")
        download_image(logo_url, tmp_path)
        convert_any_image_to_png(tmp_path, out_png_path)
    except Exception as e:
        print(f"[logo] failed for url={logo_url}: {e}")
        make_text_placeholder_png(out_png_path, (fallback_text or "TEAM")[:10])
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# =========================================================
# ffprobe helpers
# =========================================================

def ffprobe_duration_ms(path: str) -> int:
    """
    Read media duration in milliseconds using ffprobe.
    Return 0 on failure.
    """
    cmd = [
        "ffprobe",
        "-v", FFPROBE_LOGLEVEL,
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        path
    ]
    p = run_cmd(cmd, check=False)

    if p.returncode != 0:
        log(f"[ffprobe_duration_ms] failed for {path}")
        if DEBUG and getattr(p, "stderr", None):
            log(p.stderr)
        return 0

    try:
        return int(float(p.stdout.strip()) * 1000.0)
    except Exception:
        log(f"[ffprobe_duration_ms] bad output for {path}: {p.stdout!r}")
        return 0


def ffprobe_has_audio(path: str) -> bool:
    """
    Check whether a media file contains at least one audio stream.
    """
    cmd = [
        "ffprobe",
        "-v", FFPROBE_LOGLEVEL,
        "-select_streams", "a:0",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        path
    ]
    p = run_cmd(cmd, check=False)

    if p.returncode != 0:
        log(f"[ffprobe_has_audio] failed for {path}")
        if DEBUG and getattr(p, "stderr", None):
            log(p.stderr)

    return p.returncode == 0 and bool((p.stdout or "").strip())


def ffprobe_stream_info(path: str):
    """
    Print basic stream and container information for debugging.
    Return parsed JSON on success, otherwise None.
    """
    cmd = [
        "ffprobe",
        "-v", FFPROBE_LOGLEVEL,
        "-show_entries", "stream=index,codec_name,width,height,bit_rate",
        "-show_entries", "format=duration,bit_rate",
        "-of", "json",
        path
    ]
    p = run_cmd(cmd, check=False)

    if p.returncode != 0:
        log(f"[ffprobe] failed for {path}")
        if DEBUG and getattr(p, "stderr", None):
            log(p.stderr)
        return None

    try:
        info = json.loads(p.stdout)
        log("[ffprobe]", json.dumps(info, ensure_ascii=False, indent=2))
        return info
    except Exception:
        log("[ffprobe] raw:", p.stdout)
        return None


def ffprobe_video_fps(path: str) -> float:
    """
    Read video frame rate using ffprobe.
    Return 0.0 on failure
    """

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=nw=1:nk=1",
        path
    ]
    result = subprocess.run(cmd, capture_output = True, text = True)
    if result.returncode != 0:
        return 0.0
    
    frame_rate = result.stdout.strip()
    try:
        if "/" in frame_rate:
            numerator, denominator = frame_rate.split("/", 1)
            numerator = float(numerator)
            denominator = float(denominator)
            return 0.0 if denominator == 0 else numerator / denominator
        return float(frame_rate)
    except Exception:
        return 0.0


# =========================================================
# HLS helpers
# =========================================================

def is_m3u8_url(url: str) -> bool:
    """
    Check whether a URL points to an HLS playlist.
    """
    return ".m3u8" in (url or "").lower()


def read_text_url(url: str, timeout=20) -> str:
    """
    Download text content from a URL.
    """
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return raw.decode("utf-8", errors="ignore")


# urlopen imported lazily to keep imports minimal
from urllib.request import urlopen  # noqa: E402


def parse_hls_master(master_url: str):
    """
    Parse an HLS master playlist and return:
    - audio renditions
    - video variants
    """
    txt = read_text_url(master_url, timeout=20)
    lines = [x.strip() for x in txt.splitlines() if x.strip()]

    media_audio = []
    variants = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("#EXT-X-MEDIA:") and "TYPE=AUDIO" in line:
            group_m = re.search(r'GROUP-ID="([^"]+)"', line)
            name_m = re.search(r'NAME="([^"]+)"', line)
            uri_m = re.search(r'URI="([^"]+)"', line)
            default_m = re.search(r'DEFAULT=([^,]+)', line)
            autoselect_m = re.search(r'AUTOSELECT=([^,]+)', line)

            media_audio.append({
                "group_id": group_m.group(1) if group_m else None,
                "name": name_m.group(1) if name_m else None,
                "uri": urljoin(master_url, uri_m.group(1)) if uri_m else None,
                "default": (default_m.group(1).strip().upper() == "YES") if default_m else False,
                "autoselect": (autoselect_m.group(1).strip().upper() == "YES") if autoselect_m else False,
            })

        elif line.startswith("#EXT-X-STREAM-INF:"):
            attrs = line

            bw_m = re.search(r'BANDWIDTH=(\d+)', attrs)
            avg_bw_m = re.search(r'AVERAGE-BANDWIDTH=(\d+)', attrs)
            res_m = re.search(r'RESOLUTION=(\d+)x(\d+)', attrs)
            audio_m = re.search(r'AUDIO="([^"]+)"', attrs)

            bw = int(bw_m.group(1)) if bw_m else 0
            avg_bw = int(avg_bw_m.group(1)) if avg_bw_m else bw
            res_w = int(res_m.group(1)) if res_m else 0
            res_h = int(res_m.group(2)) if res_m else 0
            audio_group = audio_m.group(1) if audio_m else None

            j = i + 1
            while j < len(lines) and lines[j].startswith("#"):
                j += 1

            if j < len(lines):
                uri = lines[j]
                variants.append({
                    "bw": bw,
                    "avg_bw": avg_bw,
                    "res_w": res_w,
                    "res_h": res_h,
                    "audio_group": audio_group,
                    "url": urljoin(master_url, uri),
                })
            i = j

        i += 1

    return {
        "audio_renditions": media_audio,
        "variants": variants,
    }


# regex needed for HLS parsing
import re  # noqa: E402


def select_best_hls_variant_and_audio(master_url: str):
    """
    Select the best HLS video variant and matching audio rendition.

    Preference:
    - highest average bandwidth
    - highest bandwidth
    - highest resolution
    """
    parsed = parse_hls_master(master_url)
    variants = parsed["variants"]
    audio_renditions = parsed["audio_renditions"]

    if not variants:
        log("[hls] not a master playlist or no variants found")
        return {
            "video_url": master_url,
            "audio_url": None,
        }

    best = max(
        variants,
        key=lambda v: (v["avg_bw"], v["bw"], v["res_h"], v["res_w"])
    )

    audio_url = None
    group_id = best.get("audio_group")

    if group_id:
        candidates = [a for a in audio_renditions if a.get("group_id") == group_id and a.get("uri")]
        if candidates:
            candidates.sort(key=lambda a: (a["default"], a["autoselect"]), reverse=True)
            audio_url = candidates[0]["uri"]

    log(
        "[hls] selected video "
        f"avg_bw={best['avg_bw']} bw={best['bw']} "
        f"res={best['res_w']}x{best['res_h']} "
        f"audio_group={best['audio_group']} "
        f"video_url={best['url']} "
        f"audio_url={audio_url}"
    )

    return {
        "video_url": best["url"],
        "audio_url": audio_url,
    }


# =========================================================
# ffmpeg download + concat
# =========================================================

def ffmpeg_download(url: str, out_path: str):
    """
    Download a clip using ffmpeg.

    If the URL is HLS (.m3u8), try to:
    - select the best video variant
    - pair it with a separate audio rendition if available

    Otherwise, download directly from the input URL.
    """
    ensure_dir(os.path.dirname(out_path) or ".")

    if is_m3u8_url(url):
        try:
            sel = select_best_hls_variant_and_audio(url)
            video_url = sel["video_url"]
            audio_url = sel["audio_url"]

            if audio_url:
                cmd = FFMPEG_BASE + [
                    "-user_agent", "Mozilla/5.0",
                    "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
                    "-i", video_url,
                    "-user_agent", "Mozilla/5.0",
                    "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
                    "-i", audio_url,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c", "copy",
                    out_path
                ]
            else:
                log("[hls] no separate audio rendition found, falling back to single-input download")
                cmd = FFMPEG_BASE + [
                    "-user_agent", "Mozilla/5.0",
                    "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
                    "-i", video_url,
                    "-map", "0:v:0",
                    "-map", "0:a:0?",
                    "-c", "copy",
                    out_path
                ]

            run(cmd)
            print(f"[download] wrote {out_path}")
            ffprobe_stream_info(out_path)
            return

        except Exception as e:
            log(f"[hls] separate video/audio selection failed, falling back to original url: {e}")

    cmd = FFMPEG_BASE + [
        "-user_agent", "Mozilla/5.0",
        "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
        "-i", url,
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c", "copy",
        out_path
    ]
    run(cmd)

    print(f"[download] wrote {out_path}")
    ffprobe_stream_info(out_path)


def try_concat_copy(files, out_path) -> bool:
    """
    Try to concatenate media files without re-encoding using ffmpeg concat demuxer.
    This only works if codecs/parameters are compatible.
    Return True on success, False otherwise.
    """
    tmp_dir = os.path.join(os.path.dirname(out_path) or ".", "_concat_tmp_copy")
    ensure_dir(tmp_dir)
    concat_list = os.path.join(tmp_dir, "concat_list.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = FFMPEG_BASE + [
        "-f", "concat", "-safe", "0", "-i", concat_list,
        "-c", "copy",
        out_path
    ]
    p = run_cmd(cmd, check=False)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return p.returncode == 0


def ffmpeg_concat_reencode(files, out_path):
    """
    Concatenate media files by re-encoding them.
    This is slower, but works even if stream parameters differ.
    """
    tmp_dir = os.path.join(os.path.dirname(out_path) or ".", "_concat_tmp")
    ensure_dir(tmp_dir)
    concat_list = os.path.join(tmp_dir, "concat_list.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = FFMPEG_BASE + [
        "-f", "concat", "-safe", "0", "-i", concat_list,
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", VIDEO_CODEC, "-preset", VIDEO_PRESET, "-crf", VIDEO_CRF,
        "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE,
        "-movflags", "+faststart",
        out_path
    ]
    run(cmd)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def concat_with_copy_fallback(files, out_path):
    """
    Try fast stream-copy concat first.
    If it fails, re-encode the concatenation.
    """
    ok = try_concat_copy(files, out_path)
    if not ok:
        ffmpeg_concat_reencode(files, out_path)

# =========================================================
# Transitions
# =========================================================

def xfade_sequence(clips, out_path, transition_sec, transition_type):
    """
    Crossfade a sequence of clips into one final video in a single ffmpeg run
    """
    if not clips:
        raise ValueError("No clips to xfade.")
    
    ensure_dir(os.path.dirname(out_path) or ".")

    if len(clips) == 1:
        ffmpeg_concat_reencode(clips, out_path)
        return

    durations_sec = [ffprobe_duration_ms(clip_path) / 1000.0 for clip_path in clips]
    for clip_path, duration_sec in zip(clips, durations_sec):
        if duration_sec <= 0:
            raise ValueError(f"Could not read duration for {clip_path}")

    cmd = list(FFMPEG_BASE)
    for clip_path in clips:
        cmd += ["-i", clip_path]

    fc_parts = []

    first_offset = max(0.0, durations_sec[0] - transition_sec)
    fc_parts.append(
        f"[0:v][1:v]xfade=transition={transition_type}:duration={transition_sec}:offset={first_offset:.3f}[v1]"
    )
    fc_parts.append(
        f"[0:a][1:a]acrossfade=d={transition_sec}[a1]"
    )

    running_duration = durations_sec[0] + durations_sec[1] - transition_sec

    for i in range(2, len(clips)):
        offset = max(0.0, running_duration - transition_sec)
        fc_parts.append(
            f"[v{i-1}][{i}:v]xfade=transition={transition_type}:duration={transition_sec:.3f}:offset={offset:.3f}[v{i}]"
        )
        fc_parts.append(
            f"[a{i-1}][{i}:a]acrossfade=d={transition_sec:.3f}[a{i}]"
        )
        running_duration += durations_sec[i] - transition_sec

    last_idx = len(clips) - 1
    filter_complex = ";".join(fc_parts)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[v{last_idx}]",
        "-map", f"[a{last_idx}]",
        "-c:v", VIDEO_CODEC, "-preset", VIDEO_PRESET, "-crf", VIDEO_CRF,
        "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE,
        "-ar", AUDIO_RATE,
        "-movflags", "+faststart",
        out_path,
    ]
    run(cmd)


# =========================================================
# Intro + per-goal rendering
# =========================================================

def fit_text(text, max_len=18):
    """
    Shorten long text so it does not break the layout.
    """
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 3].rstrip() + "..."


def make_intro(
    out_path,
    background_img,
    home_logo,
    away_logo,
    home_name,
    away_name,
    match_name,
    fps,
    dur_sec=3.0
):
    """
    Render intro slate on prepared background:
    - custom background image
    - both logos placed into circular slots
    - team names on bottom side plates
    - match date in the center plate
    - silent audio track
    """
    fontarg = get_drawtext_font_arg()

    # auto-fit / truncate texts
    hn = escape_drawtext(fit_text(home_name, 18))
    an = escape_drawtext(fit_text(away_name, 18))
    mn = escape_drawtext(fit_text(match_name, 22))

    cmd = FFMPEG_BASE + [

        # background image looped for whole duration
        "-loop", "1",
        "-framerate", str(fps),
        "-t", str(dur_sec),
        "-i", background_img,

        # silent audio
        "-f", "lavfi",
        "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo:d={dur_sec}",

        # logos
        "-i", home_logo,
        "-i", away_logo,

        "-filter_complex",
        (
            # background
            "[0:v]scale=1920:1080[bg];"

            # logos
            "[2:v]scale=430:430[hl];"
            "[3:v]scale=430:430[al];"

            # place logos into circle slots
            "[bg][hl]overlay=x=310:y=280[tmp1];"
            "[tmp1][al]overlay=x=1210:y=280[tmp2];"

            # home team name
            f"[tmp2]drawtext={fontarg}"
            f"text='{hn}'"
            f":x=360+(380-text_w)/2"
            f":y=824"
            f":fontsize=40"
            f":fontcolor=0xF4F8FF"
            f":borderw=3:bordercolor=0x1B2F4A"
            f":shadowx=0:shadowy=2:shadowcolor=0x0A1630"
            f"[tmp3];"

            # away team name
            f"[tmp3]drawtext={fontarg}"
            f"text='{an}'"
            f":x=1180+(380-text_w)/2"
            f":y=824"
            f":fontsize=40"
            f":fontcolor=0xF4F8FF"
            f":borderw=3:bordercolor=0x1B2F4A"
            f":shadowx=0:shadowy=2:shadowcolor=0x0A1630"
            f"[tmp4];"

            # match date in center plate
            f"[tmp4]drawtext={fontarg}"
            f"text='{mn}'"
            f":x=(w-text_w)/2"
            f":y=824"
            f":fontsize=42"
            f":fontcolor=white"
            f":borderw=2:bordercolor=0x0C1A2B"
            f":shadowx=0:shadowy=2:shadowcolor=black"
            f"[vout]"
        ),

        "-map", "[vout]",
        "-map", "1:a",
        "-r", str(fps),
        "-c:v", VIDEO_CODEC, "-preset", VIDEO_PRESET, "-crf", VIDEO_CRF,
        "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE,
        "-shortest",
        "-movflags", "+faststart",
        out_path
    ]

    run(cmd)


def render_goal_clip(
    video_path,
    segments_ms,
    out_path,
    home_logo,
    away_logo,
    home_short,
    away_short,
    scorer_text,
    score_before,
    score_after,
    switch_sec,
    scoreboard_bg=None,
    scorer_bg=None,
    scorer_transition_sec=0.4,
):
    """
    Render a single goal recap clip.

    Steps:
    - trim selected intervals from the source clip
    - concatenate them in one ffmpeg graph
    - overlay scoreboard background, team logos, team short names, and score
    - show score_before before switch_sec
    - show score_after after switch_sec
    - show scorer text after switch_sec with fade-in transition
    - optionally show a background image for scoreboard
    - optionally show a background image for scorer text
    """
    if not segments_ms:
        segments_ms = [[0, 2000]]

    fontarg = get_drawtext_font_arg()
    scorer_txt = escape_drawtext(scorer_text)
    sb = escape_drawtext(score_before)
    sa = escape_drawtext(score_after)
    hs = escape_drawtext(home_short)
    aws = escape_drawtext(away_short)

    has_audio = ffprobe_has_audio(video_path)

    cmd = FFMPEG_BASE + ["-i", video_path, "-i", home_logo, "-i", away_logo]

    scoreboard_bg_input_idx = None
    scorer_bg_input_idx = None
    next_input_idx = 3

    if scoreboard_bg:
        cmd += ["-i", scoreboard_bg]
        scoreboard_bg_input_idx = next_input_idx
        next_input_idx += 1

    if scorer_bg:
         cmd += ["-loop", "1", "-i", scorer_bg]
         scorer_bg_input_idx = next_input_idx
         next_input_idx += 1

    audio_input_idx = None
    if not has_audio:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
        audio_input_idx = next_input_idx

    fc_parts = []

    for i, (a_ms, b_ms) in enumerate(segments_ms):
        start = max(0.0, a_ms / 1000.0)
        end = max(start + 0.001, b_ms / 1000.0)

        fc_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]"
        )

        if has_audio:
            fc_parts.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]"
            )
        else:
            dur = max(0.001, end - start)
            fc_parts.append(
                f"[{audio_input_idx}:a]atrim=start=0:end={dur:.3f},asetpts=PTS-STARTPTS[a{i}]"
            )
    total_duration = sum(max(0.001, (b_ms - a_ms) / 1000.0) for a_ms, b_ms in segments_ms)

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(segments_ms)))
    fc_parts.append(f"{concat_inputs}concat=n={len(segments_ms)}:v=1:a=1[cv][ca]")

    enable_before = f"lt(t\\,{switch_sec:.3f})"
    enable_after = f"gte(t\\,{switch_sec:.3f})"

    fade_dur = max(0.01, float(scorer_transition_sec))
    fade_end = switch_sec + fade_dur

    scorer_alpha_expr = (
        f"if(lt(t\\,{switch_sec:.3f})\\,0\\,"
        f"if(lt(t\\,{fade_end:.3f})\\,(t-{switch_sec:.3f})/{fade_dur:.3f}\\,1))"
    )

    short_text_style = "fontcolor=white:borderw=2:bordercolor=black"
    score_text_style = "fontcolor=white:borderw=2:bordercolor=black"
    scorer_text_style = "fontcolor=white:borderw=3:bordercolor=black"

    fc_parts.append("[1:v]scale=48:48[hl]")
    fc_parts.append("[2:v]scale=48:48[al]")

    scoreboard_base = "cv"

    if scoreboard_bg:
        fc_parts.append(f"[{scoreboard_bg_input_idx}:v]scale=-1:70[sboardbg]")
        fc_parts.append("[cv][sboardbg]overlay=x=20:y=18[tmp_sb0]")
        scoreboard_base = "tmp_sb0"

    fc_parts.append(f"[{scoreboard_base}][hl]overlay=x=36:y=28[tmp1]")
    fc_parts.append("[tmp1][al]overlay=x=390:y=28[tmp2]")

    if scoreboard_bg:
        score_base = "tmp2"
    else:
        fc_parts.append(
            "[tmp2]drawbox=x=110:y=24:w=106:h=48:color=black@0.45:t=fill[tmp3]"
        )
        score_base = "tmp3"

    fc_parts.append(
        f"[{score_base}]drawtext={fontarg}text='{hs}':"
        f"x=104:y=43:fontsize=28:{short_text_style}"
        "[tmp_hs]"
    )

    fc_parts.append(
        f"[tmp_hs]drawtext={fontarg}text='{aws}':"
        f"x=320:y=43:fontsize=28:{short_text_style}"
        "[tmp_names]"
    )

    fc_parts.append(
        f"[tmp_names]drawtext={fontarg}text='{sb}':"
        f"x=237-text_w/2:y=40:fontsize=34:{score_text_style}:"
        f"enable='{enable_before}'[tmp4]"
    )

    fc_parts.append(
        f"[tmp4]drawtext={fontarg}text='{sa}':"
        f"x=237-text_w/2:y=40:fontsize=34:{score_text_style}:"
        f"enable='{enable_after}'[tmp5]"
    )

    scorer_base = "tmp5"
    if scorer_bg:
        scorer_bg_y_expr = (
            f"if(lt(t\\,{switch_sec:.3f})\\,H+10\\,"
            f"if(lt(t\\,{fade_end:.3f})\\,"
            f"(H+10)-((t-{switch_sec:.3f})/{fade_dur:.3f})*(120)\\,"
            f"H-110))"
        )

        text_y_expr = (
            f"if(lt(t\\,{switch_sec:.3f})\\,h+20\\,"
            f"if(lt(t\\,{fade_end:.3f})\\,"
            f"(h+20)-((t-{switch_sec:.3f})/{fade_dur:.3f})*(95)\\,"
            f"h-75))"
        )

        fc_parts.append(
            f"[{scorer_bg_input_idx}:v]"
            f"fps=30,"
            f"scale=-1:100,"
            f"format=rgba,"
            f"trim=duration={total_duration:.3f},"
            f"setpts=PTS-STARTPTS"
            f"[sbg]"
        )
        fc_parts.append(
            f"[{scorer_base}][sbg]overlay="
            f"x=(W-w)/2:"
            f"y='{scorer_bg_y_expr}':"
            f"shortest=1[tmp6]"
        )
        scorer_base = "tmp6"

        fc_parts.append(
            f"[{scorer_base}]drawtext={fontarg}text='{scorer_txt}':"
            f"x=(w-text_w)/2:"
            f"y='{text_y_expr}':"
            f"fontsize=42:{scorer_text_style}"
            f"[vout]"
        )
    else:
        text_y_expr = (
            f"if(lt(t\\,{switch_sec:.3f})\\,h+20\\,"
            f"if(lt(t\\,{fade_end:.3f})\\,"
            f"(h+20)-((t-{switch_sec:.3f})/{fade_dur:.3f})*(115)\\,"
            f"h-95))"
        )

        text_alpha_expr = (
            f"if(lt(t\\,{switch_sec:.3f})\\,0\\,"
            f"if(lt(t\\,{fade_end:.3f})\\,(t-{switch_sec:.3f})/{fade_dur:.3f}\\,1))"
        )

        fc_parts.append(
            f"[{scorer_base}]drawtext={fontarg}text='{scorer_txt}':"
            f"x=(w-text_w)/2:"
            f"y='{text_y_expr}':"
            f"fontsize=46:{scorer_text_style}:"
            f"box=1:boxcolor=black@0.45:boxborderw=12:"
            f"alpha='{text_alpha_expr}'"
            f"[vout]"
        )

    filter_complex = ";".join(fc_parts)

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "[ca]",
        "-c:v",
        VIDEO_CODEC,
        "-preset",
        VIDEO_PRESET,
        "-crf",
        VIDEO_CRF,
        "-c:a",
        AUDIO_CODEC,
        "-b:a",
        AUDIO_BITRATE,
        "-ar",
        AUDIO_RATE,
        "-movflags",
        "+faststart",
        out_path,
    ]

    run(cmd)