#!/usr/bin/env python3
"""Absorption versus frequency and PSRR height / width, from psrr_size_sweep.mat.

  <out>_f_vs_height.png   A(f, height) at one fixed width   (default: grid width closest to 1.70 um)
  <out>_f_vs_width.png    A(f, width)  at one fixed height  (default: grid height closest to 1.70 um)
  PSRR height is its extent along x, width its extent along y (the gap direction).

Usage:  python plot_psrr_sweep.py [psrr_size_sweep.mat] [out_prefix] [--w 1.70] [--h 1.70] [--max] [--smooth]
  --w / --h   fixed width (um) for the height plot / fixed height (um) for the width plot
  --max       instead of a slice, show the best absorption over the other dimension at each frequency
  --smooth    interpolate between grid values (display only)
Needs: numpy, scipy, matplotlib  (+ h5py if Lumerical saved a v7.3 file:  pip install h5py)
Skipped (impossible-geometry) cells are shown grey.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def load_mat(path):
    """loadmat, falling back to h5py for MATLAB v7.3 (HDF5) files, which Lumerical writes for big data."""
    try:
        return loadmat(path)
    except (NotImplementedError, ValueError):
        import h5py
        out = {}
        with h5py.File(path, "r") as h:
            for k, v in h.items():
                if isinstance(v, h5py.Dataset):
                    out[k] = np.array(v).T      # v7.3 stores arrays with reversed dimensions
        return out


ap = argparse.ArgumentParser()
ap.add_argument("mat", nargs="?", default="psrr_size_sweep.mat")
ap.add_argument("prefix", nargs="?", default="psrr")
ap.add_argument("--w", type=float, default=1.70)
ap.add_argument("--h", type=float, default=1.70)
ap.add_argument("--max", action="store_true")
ap.add_argument("--smooth", action="store_true")
args = ap.parse_args()

d = load_mat(args.mat)
get = lambda k: np.squeeze(np.asarray(d[k]))
f = get("f_THz").astype(float)
H = get("h_vec").astype(float) * 1e6
W = get("w_vec").astype(float) * 1e6
Nh, Nw, n = len(H), len(W), len(f)
valid = np.asarray(d["valid"]).reshape(Nh, Nw) > 0
order = np.argsort(f)
f = f[order]
A = np.asarray(d["A_all"], dtype=float).reshape(n, Nh, Nw)[order]      # [f, height, width]
A[:, ~valid] = np.nan
print(f"{valid.sum()} of {Nh * Nw} grid cells have data")

h_ref, w_ref = float(get("h_ref")) * 1e6, float(get("w_ref")) * 1e6
if int(get("ref_done")) == 1:
    ra = np.asarray(d["ref_A"], dtype=float).squeeze()[order]
    i = int(np.argmax(ra))
    print(f"reference run (h = {h_ref:.2f}, w = {w_ref:.2f} um): peak A = {ra[i]:.3f} at {f[i]:.2f} THz")

cm = plt.get_cmap("magma").copy()
cm.set_bad("0.85")


def nearest_valid(vals, target, ok):
    idx = np.argsort(np.abs(vals - target))
    for i in idx:
        if ok[i]:
            return int(i)
    raise SystemExit("no valid cells yet")


def panel(axis_vals, data, ylabel, title, fname, mark=None):
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    im = ax.pcolormesh(f, axis_vals, data.T, cmap=cm, vmin=0, vmax=1,
                       shading="gouraud" if args.smooth else "nearest")
    if mark is not None:
        ax.axhline(mark, color="c", ls=":", lw=1.2, label=f"reference value, {mark:g} µm")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    step = np.diff(axis_vals).mean()
    ax.set_ylim(axis_vals[0] - step / 2, axis_vals[-1] + step / 2)
    ax.set_xlim(f.min(), f.max())
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Absorption")
    fig.savefig(fname, dpi=200)
    print("wrote", fname)


nan_slice = lambda a, axis: np.where(np.all(np.isnan(a), axis=axis), np.nan, np.nanmax(np.where(np.isnan(a), -1, a), axis=axis))

if args.max:
    # best absorption over the other dimension (NaN only if every cell on that line is blank)
    d_h = np.nanmax(np.where(np.isnan(A), -np.inf, A), axis=2)       # [f, height]
    d_h[~np.isfinite(d_h)] = np.nan
    d_w = np.nanmax(np.where(np.isnan(A), -np.inf, A), axis=1)       # [f, width]
    d_w[~np.isfinite(d_w)] = np.nan
    t_h = "Absorption vs frequency and PSRR height\n(best over all widths)"
    t_w = "Absorption vs frequency and PSRR width\n(best over all heights)"
else:
    jw = nearest_valid(W, args.w, valid.any(axis=0))
    ih = nearest_valid(H, args.h, valid.any(axis=1))
    d_h = A[:, :, jw]
    d_w = A[:, ih, :]
    t_h = f"Absorption vs frequency and PSRR height\n(width = {W[jw]:.2f} µm)"
    t_w = f"Absorption vs frequency and PSRR width\n(height = {H[ih]:.2f} µm)"

panel(H, d_h, "PSRR height (µm)", t_h, f"{args.prefix}_f_vs_height.png", mark=h_ref)
panel(W, d_w, "PSRR width (µm)", t_w, f"{args.prefix}_f_vs_width.png", mark=w_ref)
plt.show()
