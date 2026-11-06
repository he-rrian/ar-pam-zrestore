"""XZ 对比（彩色，统一 Z 尺度 1.842 um/px）：光片完整 .ims，声光/增强 10 mm。用法: python compare_full_xz2.py 4|1"""
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

MOUSE = sys.argv[1] if len(sys.argv) > 1 else "4"
IMS = rf"D:\tmp\data\{MOUSE}.ims"
PX = 1.842
LS_PX = 1.842           # 统一：光片也用 1.842
NZ = 686
Z0 = 1311 if MOUSE == "4" else 50
FLIP = (MOUSE == "4")
SLICES = r"D:\tmp\work\pam_slices.npy" if MOUSE == "4" else r"D:\tmp\work\pam_slices_1.npy"
SLICES_E = r"D:\tmp\work\enh_slices.npy" if MOUSE == "4" else r"D:\tmp\work\enh_slices_1.npy"

pam = np.load(SLICES)
enh = np.load(SLICES_E)
NZ = pam.shape[1]
frames = [150, 400, 650, 880]

f = h5py.File(IMS, "r")
ds = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
NZi, NY, NX = ds.shape


def ls_slice(fi):
    y = C.pam_frame_to_ls_row(fi) + C.LS_Y_START
    a = ds[Z0:Z0 + NZ, y, :].astype(np.float32)
    if FLIP:
        a = a[::-1]
    a = np.maximum(a - np.percentile(a, 10), 0)
    n_out = int(round(NX * 2.1 / C.LS_X_BIN_UM))
    edges = np.linspace(0, NX, n_out + 1).astype(np.int64)
    return np.add.reduceat(a, edges[:-1], axis=1) / np.diff(edges)[None, :]


fig, axs = plt.subplots(len(frames), 3, figsize=(16, 2.8 * len(frames)))
for r, fi in enumerate(frames):
    ls = ls_slice(fi)
    panels = [(f"light-sheet (full .ims) {ls.shape[1]*C.LS_X_BIN_UM/1000:.2f} mm", ls, LS_PX),
              (f"raw PAM {pam[r].shape[1]*C.LS_X_BIN_UM/1000:.2f} mm", pam[r], PX),
              (f"enhanced {enh[r].shape[1]*C.LS_X_BIN_UM/1000:.2f} mm", enh[r], PX)]
    for c, (title, img, pz) in enumerate(panels):
        ax = axs[r, c]
        ax.imshow(img, cmap="magma", aspect="auto",
                  extent=[0, img.shape[1] * C.LS_X_BIN_UM / 1000, img.shape[0] * pz / 1000, 0],
                  vmin=0, vmax=float(np.percentile(img, 99.9)))
        ax.set_title(title + f"  Y={fi*C.PAM_FRAME_PITCH_UM/1000:.2f} mm", fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        if c == 0:
            ax.set_ylabel(f"Z (mm, 1px={pz}um)", fontsize=8)
fig.suptitle(f"Mouse{MOUSE} XZ: LS from complete .ims, all panels at {PX} um/px", fontsize=12)
fig.tight_layout()
p = rf"D:\tmp\result\6\xz_samez_mouse{MOUSE}.png"
fig.savefig(p, dpi=160)
print("saved", p)
