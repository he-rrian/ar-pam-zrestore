"""放大版 XZ 三联：光片 / 原始声光 / 增强声光（每行同一物理 Y，2 mm 宽）"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io
from data_reader import LightsheetReader, load_or_compute_stats

frames = [300, 600, 850]
x0, x1 = 300, 500          # 2 mm 宽
ls_r = LightsheetReader(C.IMS_M4_NPY, stats=load_or_compute_stats(C.IMS_M4_NPY))
raw, _ = pam_io.load_cache(C.CACHE_M4_PATH)
enh = np.load(os.path.join(C.WORK_DIR, "mouse4_enhanced_v2.npy"), mmap_mode="r")

fig, axs = plt.subplots(len(frames), 3, figsize=(15, 3.2 * len(frames)))
xmm = np.array([x0, x1]) * C.LS_X_BIN_UM / 1000
for r, fi in enumerate(frames):
    ly = C.pam_frame_to_ls_row(fi)
    ls = ls_r.get_xz(ly + C.LS_Y_START, 0)[:, x0:x1]
    rw = np.asarray(raw[fi])[:, x0:x1]
    en = np.asarray(enh[fi])[:, x0:x1]
    for c, (img, t) in enumerate([(ls, f"LS Y={ly}"), (rw, f"raw PAM f{fi}"), (en, f"enhanced f{fi}")]):
        ax = axs[r, c]
        ax.imshow(img, cmap="magma", aspect="equal", extent=[xmm[0], xmm[1], 0.96, 0], vmin=0,
                  vmax=float(np.percentile(img, 99.95)))
        ax.set_title(t); ax.set_ylabel("Z (mm)"); ax.set_xlabel("X (mm)")
fig.suptitle("XZ zoom 2 mm: light-sheet | raw PAM | enhanced (no clipping applied)", fontsize=13)
fig.tight_layout()
out = os.path.join(C.QC_DIR, "xz_zoom_mouse4.png")
fig.savefig(out, dpi=180)
print("saved", out)
