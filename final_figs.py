"""最终图：完整 XZ 三联 + 完整视野 XY MIP（s114 几何，1.842 um/px）。"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io

PX = 1.842                      # s114 几何下的像素物理尺寸 (um)
raw, _ = pam_io.load_cache(C.CACHE_M4_PATH)
enh = np.load(os.path.join(C.WORK_DIR, "mouse4_enhanced_v2.npy"), mmap_mode="r")
grid = np.load(C.IMS_M4_NPY, mmap_mode="r")
NY, NZ = raw.shape[0], raw.shape[1]
x_mm = 1000 * C.LS_X_BIN_UM / 1000
z_mm = NZ * PX / 1000

frames = [150, 400, 650, 880]
fig, axs = plt.subplots(len(frames), 3, figsize=(15, 2.7 * len(frames)))
for r, fi in enumerate(frames):
    ly = C.pam_frame_to_ls_row(fi)
    panels = [("light-sheet", np.asarray(grid[:, ly, :])),
              ("raw PAM", np.asarray(raw[fi])),
              ("enhanced", np.asarray(enh[fi]))]
    for c, (name, img) in enumerate(panels):
        ax = axs[r, c]
        ax.imshow(img, cmap="magma", aspect="auto", extent=[0, x_mm, z_mm, 0], vmin=0,
                  vmax=float(np.percentile(img, 99.9)))
        ax.set_title(f"{name}  Y={fi*C.PAM_FRAME_PITCH_UM/1000:.2f} mm  f{fi}", fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        if c == 0:
            ax.set_ylabel(f"Z (mm, 1px={PX}um)", fontsize=8)
fig.suptitle(f"Mouse4 final: full {z_mm:.2f} mm depth x 10 mm, 1 px = {PX} um (scale-corrected geometry)",
             fontsize=12)
fig.tight_layout()
p1 = os.path.join(C.QC_DIR, "final_xz.png")
fig.savefig(p1, dpi=160)
print("saved", p1)

raw_m = np.zeros((NY, 1000), np.float32)
enh_m = np.zeros((NY, 1000), np.float32)
for i in range(NY):
    raw_m[i] = np.asarray(raw[i]).max(axis=0)
    enh_m[i] = np.asarray(enh[i]).max(axis=0)
ls_m = np.asarray(grid).max(axis=0)[::-1]
step = ls_m.shape[0] // NY
ls_m = ls_m[: step * NY].reshape(NY, step, -1).max(axis=1)

fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
for ax, img, t in [(axs[0], ls_m, "light-sheet XY MIP (Y flipped)"),
                   (axs[1], raw_m, "raw PAM XY MIP"),
                   (axs[2], enh_m, "enhanced XY MIP")]:
    ax.imshow(img, cmap="magma", aspect="equal", extent=[0, 10, NY * 0.01, 0], vmin=0,
              vmax=float(np.percentile(img, 99.5)))
    ax.set_title(t, fontsize=11); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
fig.tight_layout()
p2 = os.path.join(C.QC_DIR, "final_mip.png")
fig.savefig(p2, dpi=170)
print("saved", p2)
