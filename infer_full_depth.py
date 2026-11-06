"""全深度推理（998 帧 × 714 px = 1500 um），复用 best_v3，输出 + 对比图 + 指标。"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import config as C
from model import AxialNAFNet
from simulate import rl_deconv

dev = "cuda" if torch.cuda.is_available() else "cpu"
import sys
SRC = os.path.join(C.WORK_DIR, sys.argv[1] if len(sys.argv) > 1 else "4_pam_full.npy")
DST = os.path.join(C.WORK_DIR, "mouse4_full_enhanced.npy")
arr = np.load(SRC, mmap_mode="r")
N, Z, X = arr.shape
print("full cache", arr.shape)

model = AxialNAFNet(in_ch=6, base_ch=-1).to(dev)
model.load_state_dict(torch.load("best_v3.pth", map_location=dev, weights_only=True))
model.eval()

out = np.empty((N, Z, X), np.float32)
with torch.no_grad():
    for i in range(2, N - 2):
        ch = [np.asarray(arr[i + k], np.float32) for k in (-2, -1, 0, 1, 2)]
        rl = rl_deconv(ch[2], C.PSF_SIGMA_Z)[None, None]
        x = torch.from_numpy(np.stack(ch)[None]).to(dev)
        x = torch.cat([x, torch.from_numpy(rl).to(dev)], dim=1)
        out[i] = model(x)[0, 0].cpu().numpy()
        if i % 200 == 0:
            print(" ", i, flush=True)
out[:2] = out[2]
out[-2:] = out[-3]
np.save(DST, out)
print("saved", DST)

# ---- 指标（全深度）----
def gauss(x, a, mu, s, b):
    return a * np.exp(-(x - mu) ** 2 / (2 * s ** 2)) + b


def fw(col, k, half=8):
    if k < half or k > len(col) - half - 1:
        return np.nan
    y = col[k - half:k + half + 1]
    try:
        p, _ = curve_fit(gauss, np.arange(-half, half + 1, dtype=float), y,
                         p0=[y.max(), 0, 2, 0], maxfev=2000)
        if 0.5 < p[2] < 12:
            return 2.355 * p[2]
    except Exception:
        pass
    return np.nan


aw, bw = [], []
for i in range(50, N - 50, 12):
    r = np.asarray(arr[i]); e = out[i]
    for c in range(0, X, 12):
        cr = r[:, c]
        pk, _ = find_peaks(cr, prominence=max(0.05, np.percentile(cr, 99)), distance=10)
        for k in pk:
            a, b = fw(cr, k), fw(e[:, c], k)
            if np.isfinite(a) and np.isfinite(b):
                aw.append(a); bw.append(b)
a = np.array(aw); b = np.array(bw)
print(f"FULL 1500um: n={len(a)} raw {np.median(a):.2f} -> enh {np.median(b):.2f} px "
      f"({np.median(a)/max(np.median(b),1e-6):.2f}x, 变窄 {100*(b<a).mean():.0f}%)")

# ---- 图：全深度 XZ ----
frames = [200, 500, 800]
fig, axs = plt.subplots(len(frames), 2, figsize=(14, 3.0 * len(frames)))
for r_, fi in enumerate(frames):
    for c, (img, t) in enumerate([(np.asarray(arr[fi]), "raw PAM (full)"),
                                  (out[fi], "enhanced (full)")]):
        ax = axs[r_, c]
        ax.imshow(img, cmap="magma", aspect="auto", extent=[0, 10, 1.5, 0], vmin=0,
                  vmax=float(np.percentile(img, 99.9)))
        ax.set_title(f"{t}  f{fi}", fontsize=10); ax.set_xlabel("X (mm)")
        if c == 0:
            ax.set_ylabel("Z (mm)")
fig.suptitle("Mouse4 FULL depth (1500 um) XZ: raw vs enhanced", fontsize=12)
fig.tight_layout()
p1 = os.path.join(C.QC_DIR, "xd_full1500_xz.png")
fig.savefig(p1, dpi=160); print("saved", p1)

# ---- 图：全深度 XY MIP ----
def mip(o_, sl):
    m = np.zeros((sl.stop - sl.start, X), np.float32)
    for k, i in enumerate(range(sl.start, sl.stop)):
        m[k] = np.asarray(o_[i]).max(axis=0)
    return m

raw_m = mip(arr, slice(0, N)); enh_m = mip(out, slice(0, N))
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
for ax, img, t in [(axs[0], raw_m, "raw PAM XY MIP (full depth)"),
                   (axs[1], enh_m, "enhanced XY MIP (full depth)")]:
    ax.imshow(img, cmap="magma", aspect="equal", extent=[0, 10, N * 0.01, 0], vmin=0,
              vmax=float(np.percentile(img, 99.5)))
    ax.set_title(t); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
fig.tight_layout()
p2 = os.path.join(C.QC_DIR, "xd_full1500_mip.png")
fig.savefig(p2, dpi=160); print("saved", p2)
