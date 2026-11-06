"""完整视野对比：XY MIP 三联 + 完整 XZ（全深度、全宽度，不裁剪）。用法: python make_fullview.py 4|1"""
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
enh = np.load(os.path.join(C.WORK_DIR, f"mouse{MOUSE}_enhanced_v2.npy"), mmap_mode="r")
grid = np.load(GRID, mmap_mode="r")

NY = raw.shape[0]
pam_mm_y = NY * C.PAM_FRAME_PITCH_UM / 1000
x_mm = 1000 * C.LS_X_BIN_UM / 1000


def mip_pam(a):
    out = np.zeros((a.shape[0], a.shape[2]), np.float32)
    for i in range(a.shape[0]):
        out[i] = np.asarray(a[i]).max(axis=0)
    return out


raw_mip = mip_pam(raw)
enh_mip = mip_pam(enh)
ls_full = np.asarray(grid).max(axis=0)                 # (4700, 1000)
step = ls_full.shape[0] // NY
ls_mip = ls_full[: step * NY].reshape(NY, step, -1).max(axis=1)

fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
for ax, img, t in [(axs[0], ls_mip, "light-sheet XY MIP (full)"),
                   (axs[1], raw_mip, "raw PAM XY MIP (full)"),
                   (axs[2], enh_mip, "enhanced XY MIP (full)")]:
    ax.imshow(img, cmap="magma", aspect="equal", extent=[0, x_mm, pam_mm_y, 0],
              vmin=0, vmax=float(np.percentile(img, 99.5)))
    ax.set_title(t, fontsize=11)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
fig.tight_layout()
out1 = os.path.join(C.QC_DIR, f"mip_full_mouse{MOUSE}.png")
fig.savefig(out1, dpi=170)
print("saved", out1)

frames = [NY // 5, NY // 2, int(NY * 0.8)]
fig, axs = plt.subplots(len(frames), 3, figsize=(16, 3.0 * len(frames)))
for r, fi in enumerate(frames):
    ly = C.pam_frame_to_ls_row(fi)
    panels = [("light-sheet", np.asarray(grid[:, ly, :])),
              ("raw PAM", np.asarray(raw[fi])),
              ("enhanced", np.asarray(enh[fi]))]
    for c, (name, img) in enumerate(panels):
        ax = axs[r, c]
        ax.imshow(img, cmap="magma", aspect="auto",
                  extent=[0, x_mm, img.shape[0] * C.LS_PITCH_UM / 1000, 0],
                  vmin=0, vmax=float(np.percentile(img, 99.9)))
        ax.set_title(f"{name} XZ full depth  f{fi}", fontsize=10)
        ax.set_xlabel("X (mm)")
        if c == 0:
            ax.set_ylabel("Z from window top (mm)")
fig.suptitle(f"Mouse{MOUSE} XZ: full 960 um depth x 10 mm width (Z stretched)", fontsize=12)
fig.tight_layout()
out2 = os.path.join(C.QC_DIR, f"xz_fullframe_mouse{MOUSE}.png")
fig.savefig(out2, dpi=170)
print("saved", out2)
