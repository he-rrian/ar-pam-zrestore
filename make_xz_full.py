"""完整 XZ 对比：按表面下深度对齐，3mm 宽 × 0.67mm 深，声光/增强共用同一表面。"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io
import surface as S
from data_reader import LightsheetReader, load_or_compute_stats

DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 320
XSPAN = int(sys.argv[2]) if len(sys.argv) > 2 else 300
X0 = int(sys.argv[3]) if len(sys.argv) > 3 else 300
frames = [150, 350, 550, 750]

ls_r = LightsheetReader(C.IMS_M4_NPY, stats=load_or_compute_stats(C.IMS_M4_NPY))
raw, _ = pam_io.load_cache(C.CACHE_M4_PATH)
enh = np.load(os.path.join(C.WORK_DIR, "mouse4_enhanced_v2.npy"), mmap_mode="r")


def surface_row(img, depth):
    a = np.asarray(img, np.float32)
    pia = S.pia_from_ls(a)
    ok = np.isfinite(pia) & (pia > 3) & (pia < a.shape[0] - depth - 3)
    if ok.sum() > 0.2 * len(pia):
        return int(np.median(pia[ok]))
    return None


def crop(img, s, depth):
    a = np.asarray(img, np.float32)
    top = max(0, (s if s is not None else 0) - 5)
    bot = min(a.shape[0], top + depth)
    if bot - top < depth:
        top = max(0, bot - depth)
    return a[top:bot, X0:X0 + XSPAN]


fig, axs = plt.subplots(len(frames), 3, figsize=(12, 2.9 * len(frames)))
for r, fi in enumerate(frames):
    ly = C.pam_frame_to_ls_row(fi)
    raw_i = np.asarray(raw[fi])
    s_pam = surface_row(raw_i, DEPTH)
    s_ls = surface_row(ls_r.get_xz(ly + C.LS_Y_START, 0), DEPTH)
    s_common = s_pam or s_ls or 80
    s_pam = s_pam or s_common
    s_ls = s_ls or s_common
    panels = [("light-sheet", ls_r.get_xz(ly + C.LS_Y_START, 0), s_ls),
              ("raw PAM", raw_i, s_pam),
              ("enhanced", np.asarray(enh[fi]), s_pam)]
    for c, (name, img, s) in enumerate(panels):
        sub = crop(img, s, DEPTH)
        ax = axs[r, c]
        wmm = sub.shape[1] * C.LS_X_BIN_UM / 1000
        dmm = sub.shape[0] * C.LS_PITCH_UM / 1000
        ax.imshow(sub, cmap="magma", aspect="equal", extent=[0, wmm, dmm, 0],
                  vmin=0, vmax=float(np.percentile(sub, 99.9)))
        ax.set_title(f"{name}  f{fi}  surface row {s}", fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        if c == 0:
            ax.set_ylabel("depth below surface (mm)", fontsize=8)
fig.suptitle(f"Mouse4 XZ aligned by surface ({DEPTH*C.LS_PITCH_UM:.0f} um deep, {XSPAN*C.LS_X_BIN_UM/1000:.1f} mm wide)",
             fontsize=12)
fig.tight_layout()
out = os.path.join(C.QC_DIR, "xz_full_mouse4.png")
fig.savefig(out, dpi=170)
print("saved", out, "surfaces:", [(fi, surface_row(np.asarray(raw[fi]), DEPTH)) for fi in frames])
