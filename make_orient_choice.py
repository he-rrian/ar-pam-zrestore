"""光片 MIP 原方向 vs Y 翻转，与声光 MIP 并排，让用户直接判断哪个对。"""
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

ext = [0, 10, NY * C.PAM_FRAME_PITCH_UM / 1000, 0]
fig, axs = plt.subplots(1, 4, figsize=(22, 6.2))
for ax, img, t in [(axs[0], ls_ds, "LS MIP (native)"),
                   (axs[1], ls_ds[::-1], "LS MIP (Y flipped)"),
                   (axs[2], pam_mip, "raw PAM MIP"),
                   (axs[3], np.asarray(np.load(os.path.join(C.WORK_DIR, f"mouse{MOUSE}_enhanced_v2.npy"), mmap_mode="r")).max(axis=1), "enhanced MIP")]:
    ax.imshow(img, cmap="magma", extent=ext, aspect="equal", vmin=0,
              vmax=float(np.percentile(img, 99.5)))
    ax.set_title(t, fontsize=12); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
fig.suptitle(f"Mouse{MOUSE}: which light-sheet Y direction matches the PAM?", fontsize=14)
fig.tight_layout()
out = os.path.join(C.QC_DIR, f"orient_choice_mouse{MOUSE}.png")
fig.savefig(out, dpi=160)
print("saved", out)
