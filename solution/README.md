# Man I Love Flick - More Than a Hack 2027, Phase 1

Possession labeller for the BAR-ATH tactical recording: one `frame,label` row per video
frame (H / A / D / X), 199,319 rows. Rule-based state machine over the provided ATD
tracking, with the provided automatic Smart Tagging as a soft prior. No learning, no
ground truth, no video. See TECHNICAL.md for the method and its limitations.

## Reproduce

```bash
pip install -r solution/requirements.txt
python solution/run_pipeline.py   # writes solution/submission.csv
python evaluation/validate_submission.py solution/submission.csv
```

Requires the data bundle unpacked under `data/` as DATA.md describes (the bundle is not
part of this repository and is not redistributable). Runtime about 30 seconds, standard
library plus numpy only.
