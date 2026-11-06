"""全深度（RF 0:500 = 1500 um）小批缓存，用于验证模型能否覆盖整个 txt。"""
import glob
import os

import numpy as np
from scipy.signal import hilbert

import config as C
import pam_io

OUT = os.path.join(C.WORK_DIR, "4_pam_full_demo.npy")
FRAMES = list(range(500, 512))          # 12 帧
N_PX = int(round(500 * C.PAM_RF_PITCH_UM / C.LS_PITCH_UM))   # 714 px

files = pam_io.list_frames(C.PAM_M4_TXT_DIR)
ref = np.loadtxt(files[FRAMES[len(FRAMES) // 2]])
gain = 1.0 / (np.percentile(np.abs(ref), 99.9) + 1e-6)

arr = np.zeros((len(FRAMES), N_PX, 1000), np.float32)
for k, fi in enumerate(FRAMES):
    rf = pam_io.read_rf(files[fi])
    # read_rf 返回 (samples=500, A-lines=1000)，全用，不再切
    env = np.abs(hilbert(rf, axis=1)).astype(np.float32)
    bg = np.median(env)
    env = np.maximum(env - bg, 0)
    # 沿 Z 线性插值到 N_PX（3 um/采样 -> 2.1 um/px）
    z_src = np.arange(env.shape[0]) * C.PAM_RF_PITCH_UM
    z_dst = np.arange(N_PX) * C.LS_PITCH_UM
    e2 = env if env.shape[0] == 500 else env.T   # -> (samples, A-lines)
    out = np.empty((N_PX, e2.shape[1]), np.float32)
    for i in range(e2.shape[1]):
        out[:, i] = np.interp(z_dst, z_src, e2[:, i])
    arr[k] = out * gain

np.save(OUT, arr)
print("saved", OUT, arr.shape, "mean", float(arr.mean()), "max", float(arr.max()))
