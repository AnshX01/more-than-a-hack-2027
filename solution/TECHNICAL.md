# Technical notes - More Than a Hack 2027, Phase 1 submission

Team: Man I Love Flick. Task: label every frame of the tactical recording of BAR-ATH as
H (home possession), A (away possession), D (dead ball) or X (ungraded) - 199,319 rows.

## Method at a glance

A rule-based state machine over the provided tracking (ATD) plus the provided automatic
Smart Tagging as a soft prior. No learning, no ground truth, no video. Every constant is
chosen a priori from published sources (each constant's source is listed below).

Pipeline stages:

1. **Parsing.** ATD text + XML metadata to per-frame arrays: ball position, player
   positions per track slot, slot-to-team map. Missing observations are NaN and stay NaN.

2. **Derived ball speed.** The ATD speed column is sparsely populated, so speed is derived
   from ball positions over a ~0.5 s span. Single-frame differences at 25 fps are unusable:
   published tracking noise (~0.5 m RMSE per position) would show up as ~20 m/s of jitter.
   Over half a second that jitter averages down to ~1.7 m/s, far below the control
   threshold.

3. **Control evidence.** A team has control evidence on a frame when one of its tracked
   players is within R_CONTROL (2.0 m) of the ball while the ball is slower than V_CONTROL
   (10 m/s, roughly elite sprint speed). Both teams near the ball means contested: no
   evidence. An unobservable ball speed is not evidence against control, because the
   challenge definition says possession persists through missing-ball phases.

4. **Dead-ball evidence.** Two channels, per IFAB Law 9: the ball is out of play the
   instant it wholly crosses a line (out-of-bounds, sustained 0.4 s), and a placed ball
   (displacement under 2.0 m across 1.5 s - displacement, not speed, so jitter cancels).
   Missing ball observations forward-fill the last observation for up to 15 s: mid-play
   blackouts carry an in-play observation so play persists through occlusion, while a
   dead observation survives the retrieval blackout after a stoppage. Dead stretches
   shorter than 1.5 s between play are smoothed away; the fastest real restart takes far
   longer. Inside a tagged set-piece window, a missing ball is always dead (retrieval).

5. **Smart Tagging prior.** The provided automatic annotation (638 phase and set-piece
   intervals, team-tagged) becomes a per-frame possession prior: attacking codes imply the
   tagged team, defending codes imply their opponent, conflicting overlaps imply nobody.
   Code semantics were verified mechanically against ball-position thirds, not against any
   ground truth. Set pieces additionally mark restart windows.

6. **Possession state machine.** The owner persists through loose balls, contests and
   occlusions (the challenge's definition). The challenger accumulates exclusive control
   evidence; 3 sustained frames switch possession, while owner contact only erodes the
   accumulator by one, so a chasing defender's stray proximity cannot erase an established
   change but a real contest still blocks it. Where tracking is silent in a trailing 10 s
   window, a sustained conflicting tagging prior (40% of the window, 3-to-1 dominance) can
   switch ownership instead; this is what covers the first 33 minutes of half 1, where the
   tactical tracking contains no player positions at all.

7. **Restart attribution.** A stoppage clears the owner. The restart belongs to whoever
   establishes control first - the tagged set-piece taker when there is one - and the gap
   between the restart and that establishment is attributed to that team retrospectively.

8. **Segment cleanup.** Spells shorter than 0.5 s between identical neighbours are frame
   flicker and absorbed; the published gamma model of possession duration (tau = 1.09 s)
   has real mass below 1 s, so nothing longer is touched. The published pre-kickoff stretch
   of half 2 is forced to D, and frames outside the graded blocks are X.

## Known limitations

- The tactical ATD contains no player tracking for graded frames 11185-61185 (about 33
  minutes of half 1). Possession there is tagging-prior driven with persistence, so
  possession changes inside a tagged phase cannot be resolved. This caps segmentation
  quality for about a third of the match.
- The ball is missing on ~41% of graded frames, so dead-ball detection leans on
  hysteresis; short stoppages during long blackouts can be missed, which is why the dead
  share (~26% of graded frames) sits below the ~35-40% typical of full match footage.
- Smart Tagging is automatic and contains false positives and negatives (per DATA.md); it
  is treated as soft evidence and can be overridden by tracking wherever tracking exists.
- Final segmentation: 353 possessions, mean 13.4 s, median 5.9 s, versus the published
  J-League reference of ~421 possessions per match - the residual gap is mostly the
  untracked third of the match.

## Verification

- `evaluation/validate_submission.py`: PASS (199,319 rows, label set and ranges well-formed).
- Synthetic-unit tests over the state machine (parser alignment, hysteresis, persistence,
  restart attribution) pass.
- The public 5-minute ground truth was used exactly once, pre-revision, as the permitted
  well-formedness check via the organisers' own `run_evaluation.py` (home-share error 0.034,
  alive/dead F1 0.916, and the segmentation defect that motivated the documented revision).
  It was never re-run afterwards.

## Reproducing

```
python solution/run_pipeline.py   # writes solution/submission.csv
```

Requires only numpy and the standard library. Runtime about 30 s.
