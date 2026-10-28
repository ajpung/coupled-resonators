#!/usr/bin/env python3
"""Absorption versus frequency and period, from vsrr_period_sweep.mat.

  <out>_f_vs_Px.png   A(f, Px) at the baseline Py
  <out>_f_vs_Py.png   A(f, Py) at the baseline Px

Usage:  python plot_sweep_freq.py [vsrr_period_sweep.mat] [out_prefix] [smooth]
        add the word "smooth" as 3rd argument to interpolate between periods (display only)
Needs:  numpy, scipy, matplotlib  (+ h5py if Lumerical saved a v7.3 file:  pip install h5py)
"""
import sys

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


mat_file = sys.argv[1] if len(sys.argv) > 1 else "vsrr_period_sweep.mat"
prefix = sys.argv[2] if len(sys.argv) > 2 else "sweep"
smooth = len(sys.argv) > 3 and sys.argv[3] == "smooth"
N_SI, C = 3.42, 299792458.0

d = load_mat(mat_file)
get = lambda k: np.squeeze(np.asarray(d[k]))
f = get("f_THz").astype(float)
Px = get("Px_vec").astype(float) * 1e6
Py = get("Py_vec").astype(float) * 1e6
N, n = len(Px), len(f)
valid = np.asarray(d["valid"]).reshape(N, N) > 0
ib, jb = int(get("ib")) - 1, int(get("jb")) - 1
order = np.argsort(f)
f = f[order]
A = np.asarray(d["A_all"], dtype=float).reshape(n, N, N)[order]      # [f, ix, iy]
A[:, ~valid] = np.nan


def edges(v):
    s = np.diff(v).mean()
    return np.concatenate([v - s / 2, [v[-1] + s / 2]])


def panel(per, data, fixed, which, fname):
    """data: [f, period]; plotted with frequency on x and the period on y."""
    cm = plt.get_cmap("magma").copy()
    cm.set_bad("0.85")
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    if smooth:
        im = ax.pcolormesh(f, per, data.T, cmap=cm, vmin=0, vmax=1, shading="gouraud")
    else:
        im = ax.pcolormesh(f, per, data.T, cmap=cm, vmin=0, vmax=1, shading="nearest")
    other = "y" if which == "x" else "x"
    ax.plot(C / (N_SI * per * 1e-6) / 1e12, per, "w--", lw=1.2, label=f"c/(n$_{{Si}}$·P{which}) diffraction onset")
    ax.axvline(C / (N_SI * fixed * 1e-6) / 1e12, color="c", ls=":", lw=1.2,
               label=f"c/(n$_{{Si}}$·P{other}) at P{other} = {fixed:.2f} µm")
    ax.axhline(per[ib if which == "x" else jb], color="0.7", lw=0.6, ls="-")
    ax.set_xlim(f.min(), f.max())
    ax.set_ylim(edges(per)[0], edges(per)[-1])
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(f"{which.upper()}-period P{which} (µm)")
    ax.set_title(f"Absorption vs frequency and P{which}   (P{other} = {fixed:.2f} µm"
                 f"{', interpolated' if smooth else ''})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.colorbar(im, ax=ax, label="Absorption")
    fig.savefig(fname, dpi=200)
    print("wrote", fname)


panel(Px, A[:, :, jb], Py[jb], "x", f"{prefix}_f_vs_Px.png")
panel(Py, A[:, ib, :], Px[ib], "y", f"{prefix}_f_vs_Py.png")
plt.show()
