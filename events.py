import json
import re
from urllib.request import urlopen, Request

from utils import safe_int


# Mapping match phase names to sortable numeric order.
PHASE_ORDER = {
    "1st period": 1,
    "2nd period": 2,
    "3rd period": 3,
    "overtime": 4,
    "shootout": 5,
}


def load_json_from_url(url: str):
    """
    Download and parse JSON from a remote URL.
    """
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        raw = r.read()
    return json.loads(raw.decode("utf-8"))


def load_events(path_or_url: str):
    """
    Load events JSON either from HTTP(S) URL or local file.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return load_json_from_url(path_or_url)
    with open(path_or_url, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_action_from_playlist(pl: dict):
    """
    Fallback helper:
    Some providers may store action labels under playlist.events[0].tags[0].action.
    """
    try:
        evs = pl.get("events") or []
        if not evs:
            return None
        tags = evs[0].get("tags") or []
        if not tags:
            return None
        return tags[0].get("action")
    except Exception:
        return None


def parse_events(data):
    """
    Parse raw provider event JSON into a normalized internal event list.

    Extracts:
    - action
    - video URL
    - phase
    - game time
    - score
    - scorer
    - other metadata used later in recap generation
    """
    out = []
    for e in data.get("events", []):
        tag = e.get("tag") or {}
        pl = e.get("playlist") or {}

        action = tag.get("action") or _extract_action_from_playlist(pl)
        url = pl.get("video_url")
        if not action or not url:
            continue

        phase = (tag.get("phase") or {}).get("value") or tag.get("phase") or "unknown"
        if isinstance(phase, dict):
            phase = phase.get("value") or "unknown"

        game_time = safe_int((tag.get("game_time") or {}).get("value"), tag.get("game_time"), 0)

        score = e.get("score") or tag.get("score") or pl.get("score")
        desc = pl.get("description") or tag.get("description") or ""

        scorer = None
        for k in ("scorer", "player", "scored_by", "player_name"):
            v = tag.get(k)
            if isinstance(v, dict):
                v = v.get("value")
            if v:
                scorer = str(v)
                break

        if not scorer and isinstance(desc, str):
            m = re.search(r"(scored by|goal by)\s*[:\-]?\s*(.+)$", desc, re.IGNORECASE)
            if m:
                scorer = m.group(2).strip()

        out.append({
            "id": e.get("id"),
            "action": action,
            "phase": phase,
            "phase_ord": PHASE_ORDER.get(phase, 99),
            "game_time": game_time,
            "wall_clock_time": e.get("wall_clock_time") or "",
            "score": score,
            "desc": desc,
            "duration_ms": safe_int(pl.get("duration_ms"), 0),
            "url": url,
            "scorer": scorer,
            "tag": tag,
            "playlist": pl,
        })

    out.sort(key=lambda x: (x["phase_ord"], x["game_time"], x["wall_clock_time"]))
    return out


def select_all_goals(events):
    """
    Filter only goal events and remove duplicates.
    """
    goals = [e for e in events if e["action"] == "goal"]

    seen = set()
    uniq_goals = []
    for e in goals:
        key = (
            e.get("url"),
            e.get("phase_ord"),
            e.get("game_time"),
            e.get("wall_clock_time"),
            e.get("id"),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq_goals.append(e)

    uniq_goals.sort(key=lambda x: (x["phase_ord"], x["game_time"], x["wall_clock_time"]))
    return uniq_goals


def parse_score_to_display(score_str: str):
    """
    Parse score string into:
    - home score
    - away score
    - normalized display string "A-B"

    Not heavily used later, but kept as a helper.
    """
    if not score_str:
        return "0", "0", "0-0"
    s = score_str.strip()
    s2 = s.replace(":", "-")
    parts = s2.split("-")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return parts[0], parts[1], f"{parts[0]}-{parts[1]}"
    return "0", "0", s


def parse_score_tuple(score_str):
    """
    Parse a score like '2-1' or '2:1' into a tuple (2, 1).
    Return None if parsing fails.
    """
    if not score_str:
        return None

    s = str(score_str).strip().replace(":", "-")
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m:
        return None

    return int(m.group(1)), int(m.group(2))


def format_score_tuple(t):
    """
    Format a score tuple back to a display string.
    """
    if t is None:
        return "0-0"
    return f"{t[0]}-{t[1]}"


def enrich_goal_scores(goals):
    """
    Compute score_before_display and score_after_display sequentially.

    If a provider flips score orientation for one event, try both:
    - (a, b)
    - (b, a)

    Pick the version that looks like a valid +1 progression from the previous score.
    """
    prev_after = (0, 0)

    for i, g in enumerate(goals):
        raw = parse_score_tuple(g.get("score"))
        chosen_after = None

        if raw is not None:
            candidates = [raw]
            flipped = (raw[1], raw[0])
            if flipped != raw:
                candidates.append(flipped)

            for cand in candidates:
                da = cand[0] - prev_after[0]
                db = cand[1] - prev_after[1]
                if (da, db) in [(1, 0), (0, 1)]:
                    chosen_after = cand
                    break

            if chosen_after is None:
                chosen_after = raw
        else:
            chosen_after = prev_after

        g["score_before_display"] = format_score_tuple(prev_after)
        g["score_after_display"] = format_score_tuple(chosen_after)

        prev_after = chosen_after

    return goals


def find_team_info(data):
    """
    Try to extract team names, logos, and match title from provider JSON.
    Includes a fallback image scan if dedicated logo fields are missing.
    """
    home_name = "HOME"
    away_name = "AWAY"
    home_short = "HOME"
    away_short = "AWAY"
    home_logo = None
    away_logo = None
    match_name = None

    game = None
    try:
        ev0 = (data.get("events") or [None])[0]
        if isinstance(ev0, dict):
            pl = ev0.get("playlist") or {}
            game = pl.get("game")
    except Exception:
        game = None

    if isinstance(game, dict):
        ht = game.get("home_team") or {}
        at = game.get("visiting_team") or game.get("away_team") or {}

        home_name = ht.get("name") or ht.get("short_name") or home_name
        away_name = at.get("name") or at.get("short_name") or away_name

        home_short =  ht.get("short_name") or home_name
        away_short = at.get("short_name") or away_name

        home_logo = ht.get("logo_url") or ht.get("logo") or ht.get("image")
        away_logo = at.get("logo_url") or at.get("logo") or at.get("image")

        date = game.get("date") or ""
        match_name = f""
        if date:
            match_name = f"{date}"

    if not home_logo or not away_logo:
        image_urls = []

        def walk(obj):
            if isinstance(obj, dict):
                for _, vv in obj.items():
                    walk(vv)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it)
            elif isinstance(obj, str):
                s = obj.strip()
                low = s.lower()
                if (s.startswith("http://") or s.startswith("https://")) and any(ext in low for ext in (".png", ".jpg", ".jpeg", ".webp", ".svg")):
                    image_urls.append(s)

        try:
            walk(data)
        except Exception:
            pass

        uniq = []
        seen = set()
        for u in image_urls:
            if u not in seen:
                seen.add(u)
                uniq.append(u)

        if not home_logo and len(uniq) >= 1:
            home_logo = uniq[0]
        if not away_logo and len(uniq) >= 2:
            away_logo = uniq[1]

    return {
        "home_name": home_name,
        "away_name": away_name,
        "home_short": home_short,
        "away_short": away_short,
        "home_logo": home_logo,
        "away_logo": away_logo,
        "match_name": match_name or f"",
    }
