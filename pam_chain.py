"""声光完整处理链：原始 txt 包络 → 正式缓存 → 增强，XZ 与 XY MIP 两套。"""
import glob
import os

import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io

files = pam_io.list_frames(C.PAM_M4_TXT_DIR)
print("txt 帧数", len(files))

# ---- 原始 txt：包络 + 逐帧背景，保留全部 500 采样 ----
raw = np.zeros((len(files), 500, 1000), np.float32)
for i, f in enumerate(files):
    rf = pam_io.read_rf(f)
    if pam_io.should_flip(f):
        rf = rf[:, ::-1]
    env = np.abs(hilbert(rf, axis=0)).astype(np.float32)     # (samples, A-lines)
    env = np.maximum(env - np.median(env), 0)
    raw[i] = env
    if i % 200 == 0:
        print(" ", i, flush=True)
np.save(r"D:\tmp\work\pam_raw_full.npy", raw)
print("raw", raw.shape)

cache, _ = pam_io.load_cache(rD:	mp\work_pam_full686.npy)
enh = np.load(os.path.join(C.WORK_DIR, "mouse4_enhanced_v2.npy"), mmap_mode="r")

fi = 400
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
imgs_top = [(raw[fi], "raw txt (envelope, 1500 um)", 500 * C.PAM_RF_PITCH_UM / 1000),
            (np.asarray(cache[fi]), "processed cache (1264 um)", cache.shape[1] * 1.842 / 1000),
            (np.asarray(enh[fi]), "enhanced", cache.shape[1] * 1.842 / 1000)]
for c, (img, t, zmm) in enumerate(imgs_top):
    ax = axs[0, c]
    ax.imshow(img, cmap="magma", aspect="auto", extent=[0, 10, zmm, 0], vmin=0,
              vmax=float(np.percentile(img, 99.9)))
    ax.set_title(f"{t}  f{fi}", fontsize=10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Z (mm)" if c == 0 else "")

mip_raw = raw.max(axis=1)
mip_cache = np.asarray(cache).max(axis=1)
mip_enh = np.asarray(enh).max(axis=1)
for c, (img, t) in enumerate([(mip_raw, "raw txt XY MIP"), (mip_cache, "cache XY MIP"),
                              (mip_enh, "enhanced XY MIP")]):
    ax = axs[1, c]
    ax.imshow(img, cmap="magma", aspect="equal", extent=[0, 10, len(files) * 0.01, 0], vmin=0,
              vmax=float(np.percentile(img, 99.5)))
    ax.set_title(t, fontsize=10); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)" if c == 0 else "")
fig.suptitle("Complete PAM processing chain (mouse4): raw txt -> processed -> enhanced", fontsize=12)
fig.tight_layout()
p = r"D:\tmp\result\6\pam_chain.png"
fig.savefig(p, dpi=150)
print("saved", p)
