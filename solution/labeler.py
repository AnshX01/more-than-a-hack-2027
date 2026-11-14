"""Possession labeller: frame labels H / A / D / X from ATD arrays.

Algorithm (no learning, no tuning on any ground truth):
  1. CONTROL EVIDENCE per frame: a team has control evidence when at least one
     of its tracked players is within R_CONTROL of the ball and the ball is
     slow enough to be controllable (ball speed < V_CONTROL). Frames with
     players of both teams near the ball are contested: no evidence.
  2. DEAD-BALL detection: the ball is out of play when it leaves the pitch
     (out-of-bounds evidence, IFAB Law 9) or stays stationary (static
     evidence), each sustained for its own hysteresis window. Missing ball
     observations carry the previous observation through the blackout (per
     the challenge definition, play and possession persist through
     occlusion); dead evidence is carried long enough to survive ball
     retrieval after a stoppage. Dead stretches shorter than T_MIN_DEAD
     between play are tracking noise (a real restart takes far longer) and
     are smoothed back to play.
  3. POSSESSION state machine: the owning team persists until the opposing
     team accumulates T_SWITCH seconds of control evidence without the owner
     showing evidence in between (possession persists through loose balls,
     contests and occlusions - challenge definition). A stoppage clears the
     owner; the restart belongs to the first team to establish control
     evidence afterwards, so the unowned gap between the restart and that
     establishment is attributed to that team retrospectively.
  4. SEGMENT cleanup: absorb implausibly short possessions and micro dead-ball
     flickers (a priori thresholds from the possession definition, see
     see TECHNICAL.md), then emit labels; frames outside the graded blocks are X.

Every parameter is chosen a priori from published football/physics reasoning
or the challenge's own public definition text; provenance is in TECHNICAL.md.
"""

import numpy as np

HOME, AWAY = 0, 1

PARAMS = dict(
    PITCH=(105.3, 68.0),   # metres; challenge repo games.py (published)
    FPS=25,
    R_CONTROL=2.0,         # m: player-ball distance for control evidence
    V_CONTROL=10.0,        # m/s: ball faster than elite sprint speed cannot be controlled
    T_SWITCH=0.12,         # s: opponent control evidence to change possession
                           # (3 frames of sustained exclusive proximity; single
                           #  frame deflections die on the evidence decay)
    T_OOB=0.4,             # s: ball out of bounds this long -> dead
    T_STATIC=1.5,          # s: ball stationary this long -> dead
    V_STATIC=0.5,          # m/s: ball under this speed is stationary
    T_ALIVE_EXIT=0.5,      # s: sustained alive evidence to end a dead stretch
    T_MIN_POSS=0.5,        # s: absorb possessions shorter than this
                           # (gamma possession model tau=1.09s: real possessions
                           #  exist well below 1s; 0.5s only kills frame flicker)
    T_MIN_DEAD=1.5,        # s: dead stretches shorter than this are tracking noise
)



def derived_ball_speed(arrays, p):
    """Position-derived ball speed (m/s), over a ~0.5s span. Single-frame
    differences are useless at 25fps: the published tracking position noise
    (~0.5m RMSE, PLOS One EPTS validation) would read as ~20 m/s of jitter, so the
    span must be long enough that noise (span noise ~0.8m over 0.5s ~ 1.7 m/s)
    sits well below the control threshold. NaN where no present observation
    exists 0.25-1.0s before the frame."""
    bx, by = arrays["bx"], arrays["by"]
    L, W = p["PITCH"]
    n = len(bx)
    v = np.full(n, np.nan, dtype=np.float32)
    present_idx = np.flatnonzero(~np.isnan(bx))
    span = int(p.get("SPEED_SPAN", 0.5) * p["FPS"])   # 12 frames
    lo, hi = int(0.25 * p["FPS"]), int(1.0 * p["FPS"])
    for f in present_idx:
        cand = present_idx[(present_idx >= f - hi) & (present_idx <= f - lo)]
        if len(cand) == 0:
            continue
        ref = cand[np.argmin(np.abs(cand - (f - span)))]
        dt = (f - ref) / float(p["FPS"])
        v[f] = np.hypot((bx[f] - bx[ref]) * L, (by[f] - by[ref]) * W) / dt
    return v


def static_evidence(arrays, p):
    """Ball effectively unmoving: displacement under R_STILL metres across a
    T_STATIC-second span. Displacement over a long span, not speed, because
    position jitter cancels; a genuinely placed ball (free kick, goal kick,
    corner) stays within ~1m while any live ball travels metres per second."""
    bx, by = arrays["bx"], arrays["by"]
    L, W = p["PITCH"]
    n = len(bx)
    out = np.zeros(n, dtype=bool)
    present_idx = np.flatnonzero(~np.isnan(bx))
    span = max(1, int(p["T_STATIC"] * p["FPS"]))
    slack = int(0.5 * p["FPS"])
    r = p.get("R_STILL", 2.0)
    for f in present_idx:
        cand = present_idx[(present_idx >= f - span - slack) & (present_idx <= f - span + slack)]
        if len(cand) == 0:
            continue
        ref = cand[np.argmin(np.abs(cand - (f - span)))]
        if np.hypot((bx[f] - bx[ref]) * L, (by[f] - by[ref]) * W) < r:
            out[f] = True
    return out


def control_evidence(arrays, p):
    """Per-frame team with control evidence: HOME, AWAY or -1 (none/both)."""
    px, py = arrays["px"], arrays["py"]
    bx, by = arrays["bx"], arrays["by"]
    v = derived_ball_speed(arrays, p)
    team = arrays["team"]
    L, W = p["PITCH"]
    n_frames, n_slots = px.shape

    dx = (px - bx[:, None]) * L
    dy = (py - by[:, None]) * W
    dist = np.sqrt(dx * dx + dy * dy)
    # Ball speed from positions; an unobservable speed is not evidence against
    # control (possession persists through missing ball by definition).
    slow = np.isnan(v) | (v < p["V_CONTROL"])
    near = (dist < p["R_CONTROL"]) & slow[:, None]
    home_near = (near & (team[None, :] == HOME)).any(axis=1)
    away_near = (near & (team[None, :] == AWAY)).any(axis=1)

    out = np.full(n_frames, -1, dtype=np.int8)
    out[home_near & ~away_near] = HOME
    out[away_near & ~home_near] = AWAY
    return out


def dead_evidence(arrays, p):
    """Two boolean per-frame evidence arrays: out-of-bounds (IFAB Law 9: the
    instant the ball wholly crosses the line) and stationary."""
    bx, by = arrays["bx"], arrays["by"]
    present = ~np.isnan(bx)
    oob = present & ((bx < 0.0) | (bx > 1.0) | (by < 0.0) | (by > 1.0))
    static = static_evidence(arrays, p)
    return oob, static


def _sustained(mask, min_frames):
    """Boolean array: True where mask has been True for at least min_frames
    consecutive frames up to and including this one."""
    n = len(mask)
    out = np.zeros(n, dtype=bool)
    run = 0
    for i in range(n):
        run = run + 1 if mask[i] else 0
        if run >= min_frames:
            out[i] = True
    return out


def _ffill_cap(mask, present, cap):
    """Forward-fill a boolean observation through missing frames, at most
    `cap` frames; beyond that the observation is treated as absent (False)."""
    n = len(mask)
    out = np.zeros(n, dtype=bool)
    last = False
    age = cap + 1
    for i in range(n):
        if present[i]:
            last = mask[i]
            age = 0
        else:
            age += 1
        out[i] = last and age <= cap
    return out


def _smooth_dead(dead, min_frames):
    """Dead stretches shorter than min_frames between play are tracking noise
    (a real restart takes far longer); smooth them back to play."""
    n = len(dead)
    out = dead.copy()
    i = 0
    while i < n:
        if not dead[i]:
            i += 1
            continue
        j = i
        while j < n and dead[j]:
            j += 1
        if i > 0 and j < n and (j - i) < min_frames:
            out[i:j] = False
        i = j
    return out


def label_frames(arrays, p=PARAMS, graded_blocks=None, forced_dead=None, prior=None, setpiece=None):
    """Full pipeline. Returns numpy '<U1' array of H/A/D/X per frame.

    graded_blocks: list of (first, last) inclusive video frames that are
      graded; everything outside gets X. None = all frames graded.
    forced_dead: list of (first, last) inclusive ranges forced to D
      (published facts such as the pre-kickoff stretch of half 2).
    """
    fps = p["FPS"]
    n = len(arrays["bx"])
    ev = control_evidence(arrays, p)
    oob, static = dead_evidence(arrays, p)
    present = ~np.isnan(arrays["bx"])
    cap = int(p.get("FFILL_CAP", 15.0) * fps)
    oob = _ffill_cap(oob, present, cap)
    static = _ffill_cap(static, present, cap)
    dead = _sustained(oob, max(1, int(p["T_OOB"] * fps))) | \
           _sustained(static, max(1, int(p["T_STATIC"] * fps)))
    dead = _smooth_dead(dead, max(1, int(p["T_MIN_DEAD"] * fps)))
    # Inside a tagged set-piece window a missing ball is a ball being
    # retrieved for the restart: stay dead regardless of the ffill cap.
    if setpiece is not None:
        dead = dead | (setpiece & ~present)

    labels = np.full(n, "D", dtype="<U1")
    is_alive = np.zeros(n, dtype=bool)
    alive = False
    alive_run = 0
    owner = -1
    acc = 0        # evidence frames accumulated by the challenging team
    cand = -1      # challenging team accumulating evidence
    need = max(1, int(p["T_SWITCH"] * fps))
    exit_need = max(1, int(p["T_ALIVE_EXIT"] * fps))

    for f in range(n):
        if dead[f]:
            alive = False
            alive_run = 0
            if owner != -1:
                owner = -1   # stoppage: the restart decides the next owner
            acc = 0
            cand = -1
            labels[f] = "D"
            continue

        if not alive:
            alive_run += 1
            if alive_run < exit_need:
                labels[f] = "D"
                continue
            alive = True
        is_alive[f] = True

        e = int(ev[f])
        if e != -1:
            if owner == -1:
                if e == cand:
                    acc += 1
                else:
                    cand = e
                    acc = 1
                if acc >= need:
                    owner = cand
                    acc = 0
                    cand = -1
            elif e == owner:
                acc = max(0, acc - 1)   # stray contact erodes, not erases
            else:
                if e == cand:
                    acc += 1
                else:
                    cand = e
                    acc = max(1, acc // 2 + 1)
                if acc >= need:
                    owner = cand
                    acc = 0
                    cand = -1

        pr = int(prior[f]) if prior is not None else -1
        if pr != -1:
            if owner == -1:
                owner = pr      # the tagged phase/restart names the owner
                acc = 0
                cand = -1
            elif pr != owner:
                # Tagging can overturn tracking-derived ownership only where
                # tracking is silent: require the trailing T_TAG seconds to be
                # free of tracking evidence and dominated by the other side's
                # prior (phases are tens of seconds; flicker is not).
                w = int(p.get("T_TAG", 10.0) * fps)
                lo = max(0, f - w)
                ev_n = int((ev[lo:f] != -1).sum())
                pr_n = int((prior[lo:f] != -1).sum())
                opp_n = int((prior[lo:f] == 1 - owner).sum())
                if ev_n < 10 and pr_n > 0 and opp_n >= 0.4 * w and opp_n >= 3 * (pr_n - opp_n):
                    owner = 1 - owner
                    acc = 0
                    cand = -1

        labels[f] = "H" if owner == HOME else ("A" if owner == AWAY else "D")

    # Restart attribution: an unowned stretch of play is followed by the
    # restart-taker establishing control; attribute the gap to that team.
    f = 0
    while f < n:
        if is_alive[f] and labels[f] == "D":
            j = f
            while j < n and is_alive[j] and labels[j] == "D":
                j += 1
            if j < n and is_alive[j] and labels[j] in ("H", "A"):
                labels[f:j] = labels[j]
            f = j
        else:
            f += 1

    labels = _cleanup(labels, p)

    if forced_dead:
        for first, last in forced_dead:
            labels[first:last + 1] = "D"
    if graded_blocks is not None:
        mask = np.zeros(n, dtype=bool)
        for first, last in graded_blocks:
            mask[first:last + 1] = True
        labels[~mask] = "X"
    return labels


def _segments(labels):
    """Yield (start, end_exclusive, label) runs."""
    n = len(labels)
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        yield i, j, labels[i]
        i = j


def _cleanup(labels, p):
    """Absorb micro possessions and micro dead flickers (a priori rules)."""
    fps = p["FPS"]
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        segs = list(_segments(labels))
        for k, (s, e, lab) in enumerate(segs):
            dur = (e - s) / fps
            prev_lab = segs[k - 1][2] if k > 0 else None
            next_lab = segs[k + 1][2] if k + 1 < len(segs) else None
            if lab in ("H", "A") and dur < p["T_MIN_POSS"] and prev_lab == next_lab and prev_lab in ("H", "A"):
                labels[s:e] = prev_lab
                changed = True
                break
            if lab == "D" and dur < p["T_MIN_DEAD"] and prev_lab == next_lab and prev_lab in ("H", "A"):
                labels[s:e] = prev_lab
                changed = True
                break
    return labels
