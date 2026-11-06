"""全深度(686层=1440um)完整 XZ 三联 + 全深度 MIP：光片/原始/增强，按表面下深度对齐。"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io
import surface as S

raw, _ = pam_io.load_cache(C.CACHE_M4_PATH)          # (998,686,1000)
enh = np.load(os.path.join(C.WORK_DIR, "mouse4_enhanced_v2.npy"), mmap_mode="r")
grid = np.load(C.IMS_M4_NPY, mmap_mode="r")          # (686,4700,1000)
NY = raw.shape[0]
NZ = raw.shape[1]
DEPTH = 620                                          # 表面下 620 层 = 1.30 mm
frames = [120, 380, 620, 860]
x_mm = 1000 * C.LS_X_BIN_UM / 1000


def surf(img):
    pia = S.pia_from_ls(np.asarray(img, np.float32))
    ok = np.isfinite(pia) & (pia > 3) & (pia < img.shape[0] - 60)
    return int(np.median(pia[ok])) if ok.sum() > 0.2 * len(pia) else None


def cut(img, s, depth):
    a = np.asarray(img, np.float32)
    top = max(0, (s or 0) - 6)
    bot = min(a.shape[0], top + depth)
    if bot - top < depth:
        top = max(0, bot - depth)
    return a[top:bot]


fig, axs = plt.subplots(len(frames), 3, figsize=(15, 2.6 * len(frames)))
for r, fi in enumerate(frames):
    ly = C.pam_frame_to_ls_row(fi)
    ls = np.asarray(grid[:, ly, :])
    rw = np.asarray(raw[fi])
    en = np.asarray(enh[fi])
    s_ls, s_pam = surf(ls), surf(rw)
    common = s_pam or s_ls or 80
    s_ls, s_pam = s_ls or common, s_pam or common
    d = int(max(80, min(DEPTH, ls.shape[0] - s_ls, rw.shape[0] - s_pam)))
    for c, (name, img, s) in enumerate([("light-sheet", ls, s_ls), ("raw PAM", rw, s_pam),
                                       ("enhanced", en, s_pam)]):
        sub = cut(img, s, d)
        ax = axs[r, c]
        ax.imshow(sub, cmap="magma", aspect="auto",
                  extent=[0, x_mm, sub.shape[0] * C.LS_PITCH_UM / 1000, 0], vmin=0,
                  vmax=float(np.percentile(sub, 99.9)))
        ax.set_title(f"{name}  Y={fi*C.PAM_FRAME_PITCH_UM/1000:.2f} mm  surf={s}", fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        if c == 0:
            ax.set_ylabel("depth below surface (mm)", fontsize=8)
fig.suptitle(f"Mouse4 COMPLETE XZ, full {NZ*C.LS_PITCH_UM/1000:.2f} mm depth (LS 686 layers, PAM 686 px), 10 mm wide",
             fontsize=12)
fig.tight_layout()
p1 = os.path.join(C.QC_DIR, "xz_complete_full686.png")
fig.savefig(p1, dpi=160)
print("saved", p1)

# ---- 全深度 MIP ----
raw_m = np.zeros((NY, 1000), np.float32)
enh_m = np.zeros((NY, 1000), np.float32)
for i in range(NY):
    raw_m[i] = np.asarray(raw[i]).max(axis=0)
    enh_m[i] = np.asarray(enh[i]).max(axis=0)
ls_m = np.asarray(grid).max(axis=0)
step = ls_m.shape[0] // NY
ls_m = ls_m[: step * NY].reshape(NY, step, -1).max(axis=1)
fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
for ax, img, t in [(axs[0], ls_m, "light-sheet XY MIP (686 layers)"),
                   (axs[1], raw_m, "raw PAM XY MIP (full depth)"),
                   (axs[2], enh_m, "enhanced XY MIP (full depth)")]:
    ax.imshow(img, cmap="magma", aspect="equal", extent=[0, 10, NY * 0.01, 0], vmin=0,
              vmax=float(np.percentile(img, 99.5)))
    ax.set_title(t, fontsize=11); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
fig.tight_layout()
p2 = os.path.join(C.QC_DIR, "mip_full686.png")
fig.savefig(p2, dpi=170)
print("saved", p2)
