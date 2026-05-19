# Automatisk Generering av Hockeyhøydepunkter

Dette prosjektet genererer automatisk highlight-videoer fra ishockeykamper ved hjelp av:

* event-data (målhendelser)
* videoanalyse
* SportSBD (sportsbd) for shot boundary detection
* kameravinkelklassifisering (deep learning)
* ffmpeg for rendering

Resultatet er en ferdig highlight-video med:

* intro  
* målsekvenser  
* score før/etter mål  
* scorer-informasjon  
* glatte overganger  

---

# Prosjektstruktur

```bash
VIDEO_HIGHLIGHT_GENERATION/
│
├── assets/
│   ├── arial.ttf
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
├── requirements.txt
├── README.md
```

---

# Før du kjører

## 1. Opprett virtual environment (anbefalt)

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

## 2. Installer avhengigheter

```bash
pip install -r requirements.txt
```

## 3. Installer FFmpeg

Sjekk:

```bash
ffmpeg -version
ffprobe -version
```

Hvis ikke installert → installer og legg i PATH.

---

## 4. Kameramodell

Du må ha en modell-checkpoint:

```bash
--model path/to/model.pth
```

Modellen må støtte disse klassene:

* behind_the_goal  
* close_up_player_or_field_referee  
* close_up_side_or_staff  
* main_camera_center  
* main_camera_left  
* main_camera_right  
* public_or_fans  

---

## 5. Assets

Disse brukes i rendering:

* `assets/background.png`
* `assets/scoreboard.png`
* `assets/scoredBy.png`
* `assets/arial.ttf`

---

## 6. SportSBD (sportsbd)

Brukes til:

* shot boundary detection  
* finne logo-overganger  
* strukturere videoen  

Modellen lastes automatisk første gang (krever internett).

---

# Hvordan finne `game_id`

## 1. Hent kamper i en periode

**URL anonymisert:**

```text
https://lenkeTilAPI.com/.../game?from_date=2026-03-01&to_date=2026-03-15
```

Gir liste over kamper.

---

## 2. Hent events for en kamp

**URL anonymisert:**

```text
https://lenkeTilAPI.com/.../game/<GAME_ID>/events
```

Brukes som input til pipelinen.

---

# Default highlight-lengde

## Automatisk beregning

Hvis du **ikke setter `--highlight_sec`**, brukes:

```python
highlight_sec = len(goals) * 22.0
```

### Eksempler

| Antall mål | Lengde   |
| ---------- | -------- |
| 1          | ~22 sek  |
| 3          | ~66 sek  |
| 5          | ~110 sek |

Lengden **avhenger av antall mål**

---

## Intro

* 3 sekunder reserveres til intro  
* resten brukes til klipp  

---

# Justere lengden

```bash
--highlight_sec 120
```

Overstyrer automatisk lengde.

---

# Hvordan kjøre

## Standard kjøring

**URL anonymisert:**

```bash
python3 highlights_main.py \
  --events_url "https://lenkeTilAPI.com/.../game/<GAME_ID>/events" \
  --model "/path/to/model.pth" \
  --out "output.mp4"
```

---

## Et mer detaljert eksempel

**URL anonymisert:**

```bash
python3 highlights_main.py \
  --events_url "https://lenkeTilAPI.com/.../game/<GAME_ID>/events" \
  --model "/...../modeller/best_model_resnet34.pth" \
  --out "/...../highlight_game_id_<GAME_ID>/highlights_<GAME_ID>.mp4" \
  --workdir "/...../workdir_game_id_<GAME_ID>/" \
  --highlight_sec 120 \
  --export_xlsx \
  --keep_workdir
```

Husk å sette riktig `<GAME_ID>`.

---

# Hjelp

```bash
python3 highlights_main.py --help
```

---

# Hva pipelinen gjør

1. Leser event-data  
2. Finner mål  
3. Henter score før/etter mål  
4. Laster ned videoklipp  
5. Kjører SportSBD / sportsbd 
6. Finner shots og logoer  
7. Klassifiserer kameravinkler  
8. Deler video i:  
   * core  
   * core_after  
   * replay  
9. Velger beste segmenter  
10. Renderer klipp med overlays  
11. Lager intro  
12. Slår sammen video  

---

# Roller i systemet

## SportSBD / sportsbd

* finner shot boundaries  
* finner logo-overganger  
* deler video i struktur  

**Gir struktur**

---

## Kameravinkelmodell

* klassifiserer hvert shot  
* avgjør hva som er viktig  

**Gir forståelse**

---

# Viktige argumenter

## Input

| Argument        | Beskrivelse    |
| --------------- | -------------- |
| `--events_url`  | URL til events |
| `--events_json` | Lokal JSON     |

---

## Modell

| Argument  | Beskrivelse     |
| --------- | --------------- |
| `--model` | Path til modell |

---

## Output

| Argument         | Beskrivelse       |
| ---------------- | ----------------- |
| `--out`          | Output video      |
| `--workdir`      | Midlertidig mappe |
| `--keep_workdir` | Behold filer      |

---

## Highlight

| Argument            | Beskrivelse   |
| ------------------- | ------------- |
| `--highlight_sec`   | Lengde        |
| `--transition_type` | Overgangstype |
| `--transition_sec`  | Varighet      |

---

## Segmentering

| Argument                    | Beskrivelse      |
| --------------------------- | ---------------- |
| `--min_valid_core_main_sec` | Min core-main    |
| `--tolerate_nonmain_sec`    | Toleranse        |
| `--min_segment_sec`         | Min klipplengde  |
| `--pad_ms`                  | Padding          |
| `--sbd_threshold`           | SportSBD terskel |
| `--min_gap_ms`              | Gap              |
| `--edge_guard_ms`           | Fjern støy       |
| `--core_main_nonmain_ms`    | Stop main        |

---
## Eksempler på genererte høydepunktvideoer ved hjelp av pipelinen kan ses her:
https://www.youtube.com/watch?v=fFWW_p3nbCc&list=PLYmB0x6MhzbEaTndqWwF1BemEVxiXNCH5&index=1
---