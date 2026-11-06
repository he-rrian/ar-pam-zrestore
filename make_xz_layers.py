"""6 个 Y 位置的完整 XZ 对比：按各自表面下深度对齐，全 10 mm 宽、0-840 um 深。"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io
import surface as S

MOUSE = sys.argv[1] if len(sys.argv) > 1 else "4"
CACHE = C.CACHE_M4_PATH if MOUSE == "4" else C.CACHE_M1_PATH
GRID = C.IMS_M4_NPY if MOUSE == "4" else C.IMS_M1_NPY
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 400

raw, _ = pam_io.load_cache(CACHE)
enh = np.load(os.path.join(C.WORK_DIR, f"mouse{MOUSE}_enhanced_v2.npy"), mmap_mode="r")
grid = np.load(GRID, mmap_mode="r")
NY = raw.shape[0]
frames = [int(NY * f) for f in (0.08, 0.2, 0.35, 0.5, 0.65, 0.85)]


def surf(img):
    pia = S.pia_from_ls(np.asarray(img, np.float32))
    ok = np.isfinite(pia) & (pia > 3) & (pia < img.shape[0] - 60)
    return int(np.median(pia[ok])) if ok.sum() > 0.2 * len(pia) else None


def cut(img, s, depth):
    a = np.asarray(img, np.float32)
    top = max(0, (s or 0) - 5)
    bot = min(a.shape[0], top + depth)
    if bot - top < depth:
        top = max(0, bot - depth)
    return a[top:bot]


x_mm = 1000 * C.LS_X_BIN_UM / 1000
fig, axs = plt.subplots(len(frames), 3, figsize=(15, 2.5 * len(frames)))
for r, fi in enumerate(frames):
    ly = C.pam_frame_to_ls_row(fi)
    ls = np.asarray(grid[:, ly, :])
    rw = np.asarray(raw[fi])
    s_pam = surf(rw)
    s_ls = surf(ls)
    d_ls = ls.shape[0] - (s_ls if s_ls is not None else 0) - 5
    d_pam = rw.shape[0] - (s_pam if s_pam is not None else 0) - 5
    depth = int(max(60, min(DEPTH, d_ls, d_pam)))
    for c, (name, img, s) in enumerate([("light-sheet", ls, s_ls), ("raw PAM", rw, s_pam),
                                       ("enhanced", np.asarray(enh[fi]), s_pam)]):
        sub = cut(img, s, depth)
        ax = axs[r, c]
        ax.imshow(sub, cmap="magma", aspect="auto",
                  extent=[0, x_mm, sub.shape[0] * C.LS_PITCH_UM / 1000, 0],
                  vmin=0, vmax=float(np.percentile(sub, 99.9)))
        ax.set_title(f"{name}  Y={fi*C.PAM_FRAME_PITCH_UM/1000:.2f} mm  surf={s}", fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        if c == 0:
            ax.set_ylabel("depth below surface (mm)", fontsize=8)
fig.suptitle(f"Mouse{MOUSE}: complete XZ at 6 Y positions, aligned by surface "
             f"({DEPTH*C.LS_PITCH_UM:.0f} um deep x 10 mm)", fontsize=12)
fig.tight_layout()
out = os.path.join(C.QC_DIR, f"xz_layers_mouse{MOUSE}.png")
fig.savefig(out, dpi=160)
print("saved", out)
