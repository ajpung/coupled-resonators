#!/usr/bin/env python3
"""Maps and slices from the period sweep saved by vsrr_period_sweep.lsf.

Top row    : peak absorption, peak frequency and FWHM versus (Px, Py)
Bottom row : A(f) versus Px (at the baseline Py), A(f) versus Py (at the baseline Px),
             and the baseline R, T, A spectrum
The peak is the strongest absorption maximum in the swept band; FWHM is measured around that
peak only (NaN if the half-maximum points are not both inside the band).

Usage:  python plot_period_sweep.py [vsrr_period_sweep.mat] [output.png]
Needs:  numpy, scipy, matplotlib  (+ h5py if Lumerical saved a v7.3 file:  pip install h5py)
Also writes <output>_metrics.csv next to the figure.
"""
import sys
from pathlib import Path

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


N_SI = 3.42            # approximate Si index at 6-30 um, used only for the diffraction-onset lines
PAPER_PEAK_THz = 30.21  # single-VSRR peak reported in the paper (reference marker)
C = 299792458.0

mat_file = sys.argv[1] if len(sys.argv) > 1 else "vsrr_period_sweep.mat"
out_png = Path(sys.argv[2] if len(sys.argv) > 2 else "sweep_maps.png")

d = load_mat(mat_file)


def get(name):
    return np.squeeze(np.asarray(d[name]))


f = get("f_THz").astype(float)
Px = get("Px_vec").astype(float) * 1e6        # um
Py = get("Py_vec").astype(float) * 1e6
N = len(Px)
n = len(f)
valid = np.asarray(d["valid"]).reshape(N, N) > 0
ib, jb = int(get("ib")) - 1, int(get("jb")) - 1

order = np.argsort(f)
f = f[order]


def cube(name):
    # stored as [n_points, N*N] with column (ii-1)*N + jj  ->  [f, ix, iy]
    a = np.asarray(d[name], dtype=float).reshape(n, N, N)[order]
    a[:, ~valid] = np.nan
    return a


A, R, T = cube("A_all"), cube("R_all"), cube("T_all")
n_done = int(valid.sum())
if n_done < N * N:
    print(f"note: {n_done} of {N * N} cells are complete; the rest are shown blank")


def peak_metrics(freq, a):
    """(A_max, f_peak, FWHM) for one spectrum; NaN when the spectrum is empty."""
    if np.all(np.isnan(a)):
        return np.nan, np.nan, np.nan
    i = int(np.nanargmax(a))
    amax = a[i]
    f_pk = freq[i]
    if 0 < i < len(a) - 1:                       # parabolic refinement of the peak position
        y0, y1, y2 = a[i - 1], a[i], a[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            f_pk = freq[i] + 0.5 * (y0 - y2) / den * (freq[i + 1] - freq[i])
    half = amax / 2
    j = i
    while j > 0 and a[j] >= half:
        j -= 1
    k = i
    while k < len(a) - 1 and a[k] >= half:
        k += 1
    if a[j] >= half or a[k] >= half:             # ran into the band edge
        return amax, f_pk, np.nan
    left = freq[j] + (half - a[j]) / (a[j + 1] - a[j]) * (freq[j + 1] - freq[j])
    right = freq[k - 1] + (a[k - 1] - half) / (a[k - 1] - a[k]) * (freq[k] - freq[k - 1])
    return amax, f_pk, right - left


Amax = np.full((N, N), np.nan)
Fpk = np.full((N, N), np.nan)
Fwhm = np.full((N, N), np.nan)
Amin = np.full((N, N), np.nan)
Amin[valid] = np.nanmin(A[:, valid], axis=0) if n_done else np.nan
for i in range(N):
    for j in range(N):
        Amax[i, j], Fpk[i, j], Fwhm[i, j] = peak_metrics(f, A[:, i, j])

csv_path = out_png.with_name(out_png.stem + "_metrics.csv")
with open(csv_path, "w") as fh:
    fh.write("Px_um,Py_um,A_max,f_peak_THz,FWHM_THz,A_min\n")
    for i in range(N):
        for j in range(N):
            if valid[i, j]:
                fh.write(f"{Px[i]:.4f},{Py[j]:.4f},{Amax[i, j]:.5f},{Fpk[i, j]:.4f},"
                         f"{Fwhm[i, j]:.4f},{Amin[i, j]:.5f}\n")
print("wrote", csv_path)
if n_done and np.nanmin(Amin) < -0.01:
    print(f"WARNING: A reaches {np.nanmin(Amin):.3f} in some cells; check PML / mesh / gold fit there.")


def edges(v):
    step = np.diff(v).mean()
    return np.concatenate([v - step / 2, [v[-1] + step / 2]])


def cmap(name):
    cm = plt.get_cmap(name).copy()
    cm.set_bad("0.85")
    return cm


fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), layout="constrained")

# Top row: maps vs (Px, Py)
for ax, data, name, title, lab in (
    (axs[0, 0], Amax, "magma", "Peak absorption", "A$_{max}$"),
    (axs[0, 1], Fpk, "viridis", "Peak frequency", "f$_{peak}$ (THz)"),
    (axs[0, 2], Fwhm, "cividis", "FWHM around the peak", "FWHM (THz)"),
):
    im = ax.pcolormesh(edges(Px), edges(Py), data.T, cmap=cmap(name))
    ax.plot(Px[ib], Py[jb], marker="*", ms=14, mfc="w", mec="k")
    ax.set_xlabel("Px (µm)")
    ax.set_ylabel("Py (µm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=lab, shrink=0.85)


# Bottom row: spectral slices through the baseline, with diffraction-onset lines
def slice_panel(ax, per, data, fixed_per, which):
    im = ax.pcolormesh(f, per, data.T, cmap=cmap("magma"), vmin=0, vmax=1, shading="nearest")
    ax.plot(C / (N_SI * per * 1e-6) / 1e12, per, "w--", lw=1.2, label=f"c/(n$_{{Si}}$·P{which}), swept")
    ax.axvline(C / (N_SI * fixed_per * 1e-6) / 1e12, color="c", ls=":", lw=1.2,
               label=f"c/(n$_{{Si}}$·P{'y' if which == 'x' else 'x'}), fixed")
    ax.set_xlim(f.min(), f.max())
    ax.set_ylim(edges(per)[0], edges(per)[-1])
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(f"P{which} (µm)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    return im


im = slice_panel(axs[1, 0], Px, A[:, :, jb], Py[jb], "x")
axs[1, 0].set_title(f"A(f) vs Px at Py = {Py[jb]:.2f} µm")
im = slice_panel(axs[1, 1], Py, A[:, ib, :], Px[ib], "y")
axs[1, 1].set_title(f"A(f) vs Py at Px = {Px[ib]:.2f} µm")
fig.colorbar(im, ax=axs[1, :2], label="Absorption", shrink=0.85)

ax = axs[1, 2]
ax.plot(f, R[:, ib, jb], color="#1f77b4", lw=2, label="R")
ax.plot(f, T[:, ib, jb], color="#7f7f7f", lw=2, label="T")
ax.plot(f, A[:, ib, jb], color="#d62728", lw=2, label="A")
ax.axvline(PAPER_PEAK_THz, color="k", ls=":", lw=1, label=f"paper peak, {PAPER_PEAK_THz:g} THz")
ax.set_xlim(f.min(), f.max())
ax.set_ylim(0, 1.02)
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Fraction")
ax.set_title(f"Baseline cell (Px, Py) = ({Px[ib]:.2f}, {Py[jb]:.2f}) µm")
ax.grid(alpha=0.3)
ax.legend(loc="center right", fontsize=8)
if valid[ib, jb]:
    ax.text(0.03, 0.5, f"A$_{{max}}$ = {Amax[ib, jb]:.3f} at {Fpk[ib, jb]:.2f} THz\n"
            f"FWHM = {Fwhm[ib, jb]:.2f} THz", transform=ax.transAxes, va="center", fontsize=9)

fig.savefig(out_png, dpi=200)
print("wrote", out_png)
plt.show()
