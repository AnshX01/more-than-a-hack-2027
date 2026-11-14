"""Full run: tactical ATD + Smart Tagging prior -> submission CSV. Prints only
aggregate label shares of its own output."""
import sys, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import numpy as np
from atd import load_arrays
from labeler import label_frames, PARAMS
from tagging import tagging_arrays
import importlib.util

N_FRAMES = 199319
GRADED = [(11185, 90651), (110808, 190197)]   # inclusive, from DATA.md
FORCED_DEAD = [(110808, 113741)]              # published in FAQ (pre-kickoff h2)
OUT = os.path.join(HERE, "submission.csv")

t0 = time.time()
a = load_arrays(os.path.join(REPO, "data/tactical/BAR-ATH_tactical_atd.xml"),
                os.path.join(REPO, "data/tactical/BAR-ATH_tactical_atd.txt"),
                N_FRAMES)
spec = importlib.util.spec_from_file_location(
    "lst", os.path.join(REPO, "starter", "load_smart_tagging.py"))
_lst = importlib.util.module_from_spec(spec); spec.loader.exec_module(_lst)
events = _lst.load()
prior, setpiece = tagging_arrays(events, N_FRAMES)
labels = label_frames(a, PARAMS, graded_blocks=GRADED, forced_dead=FORCED_DEAD,
                      prior=prior, setpiece=setpiece)

with open(OUT, "w") as fh:
    fh.write("frame,label\n")
    for i, lab in enumerate(labels):
        fh.write("%d,%s\n" % (i, lab))

# Aggregate-only report of MY OWN output
vals, counts = np.unique(labels, return_counts=True)
share = {v: round(100.0*c/len(labels), 2) for v, c in zip(vals, counts)}
graded_mask = np.zeros(N_FRAMES, bool)
for s, e in GRADED:
    graded_mask[s:e+1] = True
gv, gc = np.unique(labels[graded_mask], return_counts=True)
gshare = {v: round(100.0*c/graded_mask.sum(), 2) for v, c in zip(gv, gc)}
print("seconds:", round(time.time()-t0, 1))
print("all-frame shares:", share)
print("graded-frame shares:", gshare)
print("wrote", OUT)
