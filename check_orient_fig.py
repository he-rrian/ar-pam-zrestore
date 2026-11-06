"""把光片脑轮廓叠加到声光 MIP 上，同向 vs Y翻转 各一张，直接看哪个对得上。"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io

MOUSE = sys.argv[1] if len(sys.argv) > 1 else "4"
CACHE = C.CACHE_M4_PATH if MOUSE == "4" else C.CACHE_M1_PATH
GRID = C.IMS_M4_NPY if MOUSE == "4" else C.IMS_M1_NPY
raw, _ = pam_io.load_cache(CACHE)
grid = np.load(GRID, mmap_mode="r")
NY = raw.shape[0]

pam_mip = np.asarray(raw).max(axis=1)
ls_mip = np.asarray(grid).max(axis=0)
step = ls_mip.shape[0] // NY
ls_ds = ls_mip[: step * NY].reshape(NY, step, -1).max(axis=1)

def mask(m, frac=0.12):
    return (m > frac * np.percentile(m, 99)) & (m > 0)

mp, ml = mask(pam_mip), mask(ls_ds)

def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))

print(f"mouse{MOUSE} 掩膜 IoU: 同向 {iou(mp, ml):.3f}   翻转 {iou(mp, ml[::-1]):.3f}")

fig, axs = plt.subplots(1, 3, figsize=(19, 6.6))
ext = [0, 10, NY * C.PAM_FRAME_PITCH_UM / 1000, 0]
for ax, lm, t in [(axs[0], ml, "LS contour on PAM (same direction)"),
                  (axs[1], ml[::-1], "LS contour on PAM (Y flipped)"),
                  (axs[2], None, "PAM mask")]:
    ax.imshow(pam_mip, cmap="gray", extent=ext, aspect="equal",
              vmin=0, vmax=float(np.percentile(pam_mip, 99.5)))
    if lm is not None:
        ax.contour(np.linspace(0, 10, lm.shape[1]), np.linspace(0, ext[2], lm.shape[0]),
                   lm.astype(float), levels=[0.5], colors=["#00ffcc"], linewidths=1.4)
    ax.set_title(t, fontsize=11); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
fig.suptitle(f"Mouse{MOUSE}: does the light-sheet outline match the PAM in Y?", fontsize=13)
fig.tight_layout()
out = os.path.join(C.QC_DIR, f"orient_check_mouse{MOUSE}.png")
fig.savefig(out, dpi=160)
print("saved", out)
