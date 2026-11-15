#!/usr/bin/env python3
"""Absorption versus X-period and Y-period, from vsrr_period_sweep.mat.

Writes three figures next to the output name:
  <out>_peak.png   heatmap of peak absorption A_max (strongest maximum in the band), cells annotated
  <out>_fixed.png  heatmap of absorption at one frequency (default 30.21 THz), cells annotated
  <out>_grid.png   7x7 grid of A(f) spectra, one small panel per (Px, Py)

Usage:  python plot_sweep_heatmap.py [vsrr_period_sweep.mat] [out_prefix] [freq_THz]
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
prefix = sys.argv[2] if len(sys.argv) > 2 else "sweep_heatmap"
f_fixed = float(sys.argv[3]) if len(sys.argv) > 3 else 30.21

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
if (~valid).any():
    print(f"note: {valid.sum()} of {N * N} cells complete; the rest are blank")

Amax = np.nanmax(np.where(np.isnan(A), -np.inf, A), axis=0)
Amax[~valid] = np.nan
Fpk = f[np.nanargmax(np.where(np.isnan(A), -np.inf, A), axis=0)]
i_f = int(np.argmin(np.abs(f - f_fixed)))
Afix = A[i_f]


def edges(v):
    s = np.diff(v).mean()
    return np.concatenate([v - s / 2, [v[-1] + s / 2]])


def heatmap(data, title, cbar, fname, sub=None):
    cm = plt.get_cmap("magma").copy()
    cm.set_bad("0.85")
    fig, ax = plt.subplots(figsize=(7.5, 6.5), layout="constrained")
    im = ax.pcolormesh(edges(Px), edges(Py), data.T, cmap=cm, vmin=0, vmax=1)
    for i in range(N):
        for j in range(N):
            if valid[i, j] and np.isfinite(data[i, j]):
                txt = f"{data[i, j]:.2f}" if sub is None else f"{data[i, j]:.2f}\n{sub[i, j]:.1f}"
                ax.text(Px[i], Py[j], txt, ha="center", va="center", fontsize=8,
                        color="k" if data[i, j] > 0.6 else "w")
    ax.plot(Px[ib], Py[jb], marker="s", ms=26, mfc="none", mec="c", mew=2)   # baseline cell
    ax.set_xticks(Px)
    ax.set_yticks(Py)
    ax.set_xticklabels([f"{x:.2f}" for x in Px])
    ax.set_yticklabels([f"{y:.2f}" for y in Py])
    ax.set_xlabel("X-period Px (µm)")
    ax.set_ylabel("Y-period Py (µm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=cbar, shrink=0.85)
    fig.savefig(fname, dpi=200)
    print("wrote", fname)


heatmap(Amax, "Peak absorption, 10–50 THz\n(cyan box = baseline; second line = peak frequency, THz)",
        "A$_{max}$", f"{prefix}_peak.png", sub=Fpk)
heatmap(Afix, f"Absorption at {f[i_f]:.2f} THz (cyan box = baseline)", "A", f"{prefix}_fixed.png")

# Small multiples: rows = Py (top = largest), columns = Px
fig, axs = plt.subplots(N, N, figsize=(1.9 * N, 1.5 * N), sharex=True, sharey=True, layout="constrained")
for i in range(N):
    for j in range(N):
        ax = axs[N - 1 - j, i]
        if valid[i, j]:
            ax.plot(f, A[:, i, j], color="#d62728", lw=1)
        if (i, j) == (ib, jb):
            for s in ax.spines.values():
                s.set_edgecolor("c")
                s.set_linewidth(2)
        ax.set_xlim(f.min(), f.max())
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_xlabel(f"Px={Px[i]:.2f}", fontsize=7)
        if i == 0:
            ax.set_ylabel(f"Py={Py[j]:.2f}", fontsize=7)
fig.supxlabel("Frequency (THz)   [columns: Px (µm)]")
fig.supylabel("Absorption   [rows: Py (µm)]")
fig.savefig(f"{prefix}_grid.png", dpi=200)
print("wrote", f"{prefix}_grid.png")
plt.show()
