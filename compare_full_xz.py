"""XZ 对比（彩色）：光片来自完整 .ims（11.56 mm 全宽），声光/增强为 10 mm。"""
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

MOUSE = "1"
IMS = rf"D:\tmp\data\{MOUSE}.ims"
Z0, Z1 = 50, 50 + 457          # 与 686 层网格同一窗口
FLIP = False
PX = 1.842                          # 声光缓存像素 (um)
LS_PX = 2.1

pam = np.load(r"D:\tmp\work\pam_slices_1.npy")
enh = np.load(r"D:\tmp\work\enh_slices_1.npy")
frames = [150, 400, 650, 880]

f = h5py.File(IMS, "r")
ds = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
NZ, NY, NX = ds.shape
print("ims", ds.shape)


def ls_slice(fi):
    y = C.pam_frame_to_ls_row(fi) + C.LS_Y_START
    a = ds[Z0:Z1, y, :].astype(np.float32)
    if FLIP:
        a = a[::-1]
    a = np.maximum(a - np.percentile(a, 10), 0)
    # 块平均到 10 um/px（保持全宽 11.56 mm）
    n_out = int(round(NX * LS_PX / C.LS_X_BIN_UM))
    edges = np.linspace(0, NX, n_out + 1).astype(np.int64)
    s = np.add.reduceat(a, edges[:-1], axis=1) / np.diff(edges)[None, :]
    return s


fig, axs = plt.subplots(len(frames), 3, figsize=(16, 2.8 * len(frames)))
for r, fi in enumerate(frames):
    ls = ls_slice(fi)
    panels = [(f"light-sheet (full .ims)  {ls.shape[1]*C.LS_X_BIN_UM/1000:.2f} mm",
               ls, ls.shape[1] * C.LS_X_BIN_UM / 1000, LS_PX),
              (f"raw PAM  {pam[r].shape[1]*C.LS_X_BIN_UM/1000:.2f} mm",
               pam[r], pam[r].shape[1] * C.LS_X_BIN_UM / 1000, PX),
              (f"enhanced  {enh[r].shape[1]*C.LS_X_BIN_UM/1000:.2f} mm",
               enh[r], enh[r].shape[1] * C.LS_X_BIN_UM / 1000, PX)]
    for c, (title, img, wx, pz) in enumerate(panels):
        ax = axs[r, c]
        ax.imshow(img, cmap="magma", aspect="auto",
                  extent=[0, wx, img.shape[0] * pz / 1000, 0],
                  vmin=0, vmax=float(np.percentile(img, 99.9)))
        ax.set_title(title + f"   Y={fi*C.PAM_FRAME_PITCH_UM/1000:.2f} mm", fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        if c == 0:
            ax.set_ylabel("Z (mm)", fontsize=8)
fig.suptitle("Mouse1 XZ: light-sheet from complete .ims (full 11.56 mm width) vs PAM (10 mm)",
             fontsize=12)
fig.tight_layout()
p = r"D:\tmp\result\6\compare_full_xz_1.png"
fig.savefig(p, dpi=160)
print("saved", p)
