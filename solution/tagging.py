"""Smart Tagging prior: per-frame possession prior from the provided automatic
tactical annotation (Metrica's machine-generated tagging, a published input of
the challenge - see DATA.md). Treat as soft evidence, never
as ground truth (DATA.md: intervals are phases, not possessions).

Code semantics were verified mechanically against ball-position aggregates
(no GT, no video): attacking codes have the ball in the tagged team's
attacking third, defending codes in their own third.
"""

import numpy as np

ATTACKING = {
    "PROGRESSION", "BUILD UP", "CREATING CHANCES", "ATTACKING TRANSITION",
    "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX",
}
DEFENDING = {
    "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION",
}
# Tagging team names -> ATD slot teams, derived mechanically (see STATE.md):
# HOME slots defend the left goal in half 1; Barcelona attacks right in half 1.
TEAM_TO_SLOT = {"Barcelona": 0, "Atletic Club": 1}  # HOME, AWAY


def tagging_arrays(events, n_frames):
    """Returns (prior, setpiece):
    prior[f] in {HOME, AWAY, -1}: implied possessor from all overlapping
      intervals; conflicting implications cancel to -1 (contested/unknown).
    setpiece[f]: inside a SET PIECES interval (restart window)."""
    h = np.zeros(n_frames + 1, dtype=np.int32)
    a = np.zeros(n_frames + 1, dtype=np.int32)
    setpiece = np.zeros(n_frames, dtype=bool)
    for ev in events:
        s, e = ev["start_frame"], min(ev["end_frame"], n_frames)
        if s >= n_frames or e <= s:
            continue
        team = TEAM_TO_SLOT.get(ev["tags"].get("Team"), -1)
        if ev["code"] == "SET PIECES":
            setpiece[s:e] = True
            if team == -1:
                continue  # throw-ins and kickoffs carry no taker tag
            implied = team  # the taker owns the restart
        elif ev["code"] in ATTACKING and team != -1:
            implied = team
        elif ev["code"] in DEFENDING and team != -1:
            implied = 1 - team
        else:
            continue  # GOALS and untagged phases imply nothing
        (h if implied == 0 else a)[s] += 1
        (h if implied == 0 else a)[e] -= 1
    hc = np.cumsum(h)[:n_frames]
    ac = np.cumsum(a)[:n_frames]
    prior = np.full(n_frames, -1, dtype=np.int8)
    prior[(hc > 0) & (ac == 0)] = 0
    prior[(ac > 0) & (hc == 0)] = 1
    return prior, setpiece
