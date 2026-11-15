#!/usr/bin/env python3
"""Absorption versus frequency and LSRR thickness / gap, from vsrr_lsrr_sweep.mat.

  <out>_f_vs_thickness.png   A(f, thickness) at one fixed gap        (default: grid gap closest to 0.90 um)
  <out>_f_vs_gap.png         A(f, gap)       at one fixed thickness  (default: grid thickness closest to 0.05 um)

Usage:  python plot_lsrr_sweep.py [vsrr_lsrr_sweep.mat] [out_prefix] [--g 0.90] [--t 0.05] [--max] [--smooth]
  --g / --t   fixed gap (um) for the thickness plot / fixed thickness (um) for the gap plot
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
ap.add_argument("mat", nargs="?", default="vsrr_lsrr_sweep.mat")
ap.add_argument("prefix", nargs="?", default="lsrr")
ap.add_argument("--g", type=float, default=0.90)
ap.add_argument("--t", type=float, default=0.05)
ap.add_argument("--max", action="store_true")
ap.add_argument("--smooth", action="store_true")
args = ap.parse_args()

d = load_mat(args.mat)
get = lambda k: np.squeeze(np.asarray(d[k]))
f = get("f_THz").astype(float)
Tk = get("t_vec").astype(float) * 1e6
G = get("g_vec").astype(float) * 1e6
Nt, Ng, n = len(Tk), len(G), len(f)
valid = np.asarray(d["valid"]).reshape(Nt, Ng) > 0
order = np.argsort(f)
f = f[order]
A = np.asarray(d["A_all"], dtype=float).reshape(n, Nt, Ng)[order]      # [f, thickness, gap]
A[:, ~valid] = np.nan
print(f"{valid.sum()} of {Nt * Ng} grid cells have data")

t_ref, g_ref = float(get("t_ref")) * 1e6, float(get("g_ref")) * 1e6
if int(get("ref_done")) == 1:
    ra = np.asarray(d["ref_A"], dtype=float).squeeze()[order]
    i = int(np.argmax(ra))
    print(f"reference run (t = {t_ref:.2f}, g = {g_ref:.2f} um): peak A = {ra[i]:.3f} at {f[i]:.2f} THz")

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
    d_t = np.nanmax(np.where(np.isnan(A), -np.inf, A), axis=2)       # [f, thickness]
    d_t[~np.isfinite(d_t)] = np.nan
    d_g = np.nanmax(np.where(np.isnan(A), -np.inf, A), axis=1)       # [f, gap]
    d_g[~np.isfinite(d_g)] = np.nan
    t_t = "Absorption vs frequency and LSRR thickness\n(best over all gaps)"
    t_g = "Absorption vs frequency and LSRR gap\n(best over all thicknesses)"
else:
    jg = nearest_valid(G, args.g, valid.any(axis=0))
    it = nearest_valid(Tk, args.t, valid.any(axis=1))
    d_t = A[:, :, jg]
    d_g = A[:, it, :]
    t_t = f"Absorption vs frequency and LSRR thickness\n(gap = {G[jg]:.2f} µm)"
    t_g = f"Absorption vs frequency and LSRR gap\n(thickness = {Tk[it]:.2f} µm)"

panel(Tk, d_t, "LSRR thickness (µm)", t_t, f"{args.prefix}_f_vs_thickness.png", mark=t_ref)
panel(G, d_g, "LSRR gap (µm)", t_g, f"{args.prefix}_f_vs_gap.png", mark=g_ref)
plt.show()
