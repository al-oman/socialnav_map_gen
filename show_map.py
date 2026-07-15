import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("npz", help="path to saved .npz")
ap.add_argument("-o", "--out", help="output PNG (default: <name>.png)")
args = ap.parse_args()

d = np.load(args.npz)
out = args.out or os.path.splitext(os.path.basename(args.npz))[0] + ".png"

nrows, ncols = d["nav"].shape
mpp, ox, oz = float(d["mpp"]), float(d["origin_x"]), float(d["origin_z"])
plt.imshow(d["nav"], origin="lower",
           extent=[ox, ox + ncols * mpp, oz, oz + nrows * mpp])
plt.xlabel("x (m)"); plt.ylabel("z (m)")
plt.savefig(out, dpi=150)
print("saved", out, d["nav"].shape)