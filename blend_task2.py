"""
Blend saved runs from task2_autoformer_v2.py on the SAME validation split.
Usage:  !python blend_task2.py runs_D/exog_ensemble runs_E/exog_ensemble
Each argument is <out_dir>/<mode>_seed<k> or <out_dir>/<mode>_ensemble.
Prints validation metrics for each input and for their equal-weight average, and writes
blend_forecast.txt. P and E of the blend are the sums over every model it contains.
"""
import os, sys
import numpy as np
import pandas as pd

def load(spec):
    out, tag = os.path.split(spec)
    vp = np.load(os.path.join(out, f"val_{tag}.npy"))
    fc = np.array(open(os.path.join(out, f"forecast_{tag}.txt")).read().split(","), float)
    truth = np.load(os.path.join(out, "val_truth.npy"))
    if tag.endswith("_ensemble"):
        df = pd.read_csv(os.path.join(out, "ensembles.csv"))
        r = df[df["mode"] == tag[: -len("_ensemble")]].iloc[0]
    else:
        mode, seed = tag.rsplit("_seed", 1)
        df = pd.read_csv(os.path.join(out, "runs.csv"))
        r = df[(df["mode"] == mode) & (df["seed"] == int(seed))].iloc[0]
    return vp, fc, truth, int(r["params"]), int(r["epochs_run"])

def metrics(pred, true):
    err = pred - true
    return dict(val_rmse=np.sqrt((err ** 2).mean()), win_rmse=np.sqrt((err ** 2).mean(1)).mean(),
                mae=np.abs(err).mean())

parts = [load(s) for s in sys.argv[1:]]
truth = parts[0][2]
assert all(p[2].shape == truth.shape and np.allclose(p[2], truth) for p in parts), "different validation splits"
for s, p in zip(sys.argv[1:], parts):
    print(s, {k: round(float(v), 2) for k, v in metrics(p[0], truth).items()}, "P", p[3], "E", p[4])
vp = np.mean([p[0] for p in parts], 0); fc = np.mean([p[1] for p in parts], 0)
print("BLEND", {k: round(float(v), 2) for k, v in metrics(vp, truth).items()},
      "P", sum(p[3] for p in parts), "E", sum(p[4] for p in parts), "zeros", int((fc == 0).sum()))
with open("blend_forecast.txt", "w") as f:
    f.write(",".join(f"{x:.4f}" for x in fc))
print("wrote blend_forecast.txt (168 values)")
