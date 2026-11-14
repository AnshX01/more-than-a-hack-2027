"""ATD (Metrica FIFA/EPTS) loader for More Than a Hack 2027.

Reads the raw tracking text + metadata XML into dense numpy arrays indexed by
VIDEO frame (0-based, matching the submission contract: video_frame = frame_id - 1).

arrays in-process and emits only aggregates.
"""

import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """int8 array over track slots: HOME if the slot belongs to the first team
    listed in the metadata (FIFA/EPTS convention; Barcelona is home per DATA.md)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("%s: expected 2 teams" % meta_path)
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)


def load_arrays(meta_path, raw_path, n_frames):
    """Parse the raw ATD into dense arrays.

    Returns dict with:
      px, py, ps : float32 (n_frames, n_slots) player x, y (normalised), speed (m/s); NaN = missing
      bx, by, bs : float32 (n_frames,) ball x, y, speed; NaN = missing
      team       : int8 (n_slots,)
    """
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    bs = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1  # frame_id counts from 1; submission from 0
            if f < 0 or f >= n_frames:
                continue
            players = parts[1]
            if players[0] != "N":
                for slot, group in enumerate(players.split(";")):
                    if group[0] == "N":
                        continue
                    c1 = group.index(",")
                    c2 = group.index(",", c1 + 1)
                    px[f, slot] = float(group[:c1])
                    py[f, slot] = float(group[c1 + 1:c2])
                    ps[f, slot] = float(group[c2 + 1:])
            ball = parts[2]
            if ball[0] != "N":
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                bx[f] = float(ball[:c1])
                by[f] = float(ball[c1 + 1:c2])
                bs[f] = float(ball[c2 + 1:])
    return {"px": px, "py": py, "ps": ps, "bx": bx, "by": by, "bs": bs,
            "team": team}


def coverage_report(arrays):
    """Aggregate-only sanity stats. Safe to surface: counts and percentages
    over the whole match, not frame-level data."""
    bx = arrays["bx"]
    px = arrays["px"]
    return {
        "frames": int(len(bx)),
        "ball_present_pct": round(100.0 * float(np.mean(~np.isnan(bx))), 2),
        "mean_tracked_players": round(float(np.nanmean((~np.isnan(px)).sum(axis=1))), 2),
        "n_slots": int(arrays["team"].size),
        "home_slots": int((arrays["team"] == HOME).sum()),
        "away_slots": int((arrays["team"] == AWAY).sum()),
    }
