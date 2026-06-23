# Automatic Hockey Highlight Generation

This project automatically generates **ice hockey goal highlight reels** from structured game event data. The pipeline combines event metadata, broadcast video analysis, deep learning, and FFmpeg-based rendering to produce a complete highlight video.

The system uses:

* Structured event data (goal events)
* Video analysis
* SportSBD (`sportsbd`) for shot boundary and logo transition detection
* Deep learning-based camera-view classification
* FFmpeg for video processing and rendering

The generated highlight reel includes:

* Intro sequence
* Goal highlights in chronological order
* Score overlays before and after each goal
* Goal scorer information
* Team logos
* Optional replay footage
* Smooth transitions between highlights

---

# Requirements

* Python 3.10 or newer
* FFmpeg (including `ffprobe`)
* Internet connection (required on the first run to download the SportSBD model)

---

# Project Structure

```text
VIDEO_HIGHLIGHT_GENERATION/
│
├── assets/
│   ├── Arial.ttf
│   ├── background.png
│   ├── scoreboard.png
│   ├── scoredBy.png
│
├── camera_model.py
├── events.py
├── ffmpeg_utils.py
├── highlights_main.py
├── pipeline.py
├── segments.py
├── utils.py
├── validators.py
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The project depends on:

```text
opencv-python-headless==4.11.0.86
openpyxl==3.1.5
sportsbd==0.1.2
torch==2.11.0
torchvision==0.26.0
```

---

## 3. Install FFmpeg

Verify that FFmpeg is available:

```bash
ffmpeg -version
ffprobe -version
```

If FFmpeg is not installed, install it and ensure both `ffmpeg` and `ffprobe` are available in your system `PATH`.

---

# Camera Classification Model

A trained **camera-view classification model** is required.

Specify the checkpoint using:

```bash
--model path/to/model.pth
```

The model must support the following camera classes:

* behind_the_goal
* close_up_player_or_field_referee
* close_up_side_or_staff
* main_camera_center
* main_camera_left
* main_camera_right
* public_or_fans

---

# Assets

The following assets are required during rendering:

* `assets/background.png`
* `assets/scoreboard.png`
* `assets/scoredBy.png`


---

# SportSBD

SportSBD (`sportsbd`) is responsible for:

* Shot boundary detection
* Logo transition detection
* Initial video segmentation

The required model checkpoint is downloaded automatically the first time the pipeline is executed.

---

# Finding the Game ID

## 1. Retrieve Games Within a Date Range

**URL anonymized**

```text
https://example-api.com/.../game?from_date=2026-03-01&to_date=2026-03-15
```

This endpoint returns a list of available games.

---

## 2. Retrieve Events for a Game

**URL anonymized**

```text
https://example-api.com/.../game/<GAME_ID>/events
```

This endpoint returns the structured event data used by the highlight generation pipeline.

---

# Default Highlight Duration

If `--highlight_sec` is not specified, the target highlight duration is computed automatically as:

```python
highlight_sec = len(goals) * 22.0
```

### Examples

| Goals | Target Highlight Duration |
| ----: | ------------------------: |
|     1 |               ~22 seconds |
|     3 |               ~66 seconds |
|     5 |              ~110 seconds |

By default, the target highlight duration scales linearly with the number of detected goals.

A three-second intro is reserved automatically, and the remaining time budget is allocated to the goal highlights.

---

# Custom Highlight Duration

Specify a custom target duration:

```bash
--highlight_sec 120
```

This overrides the automatically calculated duration.

---

# Running the Pipeline

## Using the Events API

```bash
python3 highlights_main.py \
    --events_url "https://example-api.com/.../game/<GAME_ID>/events" \
    --model "/path/to/model.pth" \
    --out "output.mp4"
```

---

## Using a Local Events JSON File

```bash
python3 highlights_main.py \
    --events_json "events.json" \
    --model "/path/to/model.pth" \
    --out "output.mp4"
```

---

## A Longer Example

```bash
python3 highlights_main.py \
    --events_url "https://example-api.com/.../game/<GAME_ID>/events" \
    --model "/.../models/best_model_resnet34.pth" \
    --out "/.../highlight_game_<GAME_ID>/highlights.mp4" \
    --workdir "/.../workdir_<GAME_ID>/" \
    --highlight_sec 120 \
    --export_xlsx \
    --keep_workdir
```

Replace `<GAME_ID>` with the desired game identifier.

---

# Help

Display all available command-line options:

```bash
python3 highlights_main.py --help
```

---

# Pipeline Overview

The pipeline performs the following steps:

1. Read structured event data
2. Parse and order goal events
3. Compute the score before and after each goal
4. Download the corresponding video clips
5. Run SportSBD
6. Detect shot boundaries and logo transitions
7. Classify camera views
8. Split each clip into:

   * core
   * core_after
   * replay
9. Select core, supplementary, and replay segments within the available time budget
10. Render each goal clip with score and scorer overlays
11. Generate the intro sequence
12. Merge all clips into the final highlight video

---

# System Components

## SportSBD

Responsibilities:

* Detect shot boundaries
* Detect logo transitions
* Segment the broadcast into meaningful shots

**Provides the structural segmentation of the video.**

---

## Camera Classification Model

Responsibilities:

* Classify each shot according to camera view
* Identify the main gameplay, close-up, and public/fan shots used during highlight selection

**Provides semantic understanding of the broadcast content.**

---

# Command-Line Arguments

## Input

| Argument        | Description                           |
| --------------- | ------------------------------------- |
| `--events_url`  | URL to the game events endpoint       |
| `--events_json` | Local JSON file containing event data |

---

## Model

| Argument  | Description                                     |
| --------- | ----------------------------------------------- |
| `--model` | Path to the trained camera classification model |

---

## Output

| Argument         | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `--out`          | Output highlight video                                   |
| `--workdir`      | Temporary working directory                              |
| `--keep_workdir` | Preserve intermediate files                              |
| `--export_xlsx`  | Export selected segments and statistics to an Excel file |

---

## Highlight Settings

| Argument            | Description ----------------------------------------------------------------------------------------------------- |
| `--highlight_sec`   | Target highlight duration                                                     |
| `--transition_type` | FFmpeg `xfade` transition (default: `fade`); use `none` to disable transitions| 
| `--transition_sec`  | Transition duration in seconds (default: `0.5)                                |
---

## Segmentation

| Argument                    | Description                                                   | ----------------------------------------------------------------------------------------------|
| `--min_valid_core_main_sec` | Minimum duration of the detected core main-camera sequence    |
| `--tolerate_nonmain_sec`    | Maximum tolerated interruption by non-main camera views       |
| `--min_segment_sec`         | Minimum segment duration                                      |
| `--pad_ms`                  | Padding around detected logo transitions                      |
| `--sbd_threshold`           | SportSBD shot boundary detection threshold                    |
| `--min_gap_ms`              | Minimum temporal gap between detected boundaries              |
| `--edge_guard_ms`           | Trim noisy boundaries at segment edges                        |
| `--core_main_nonmain_ms`    | End the main-camera sequence after prolonged non-main footage |

---

# Output

The pipeline produces:

* Final highlight video (`.mp4`)
* Optional Excel summary (`segments.xlsx`) when `--export_xlsx` is enabled
* Temporary working directory (optional, if `--keep_workdir` is specified)

The generated highlights preserve the chronological order of goals while prioritizing continuous gameplay and supplementing it with reaction and replay footage when the available time budget allows.
