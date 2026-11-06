"""用高通结构相关评估手动对齐：identity vs 用户参数 vs 局部搜索。"""
import os

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

import config as C
import pam_io

N, mm = 600, 10.0
raw, _ = pam_io.load_cache(C.CACHE_M4_PATH)
grid = np.load(C.IMS_M4_NPY, mmap_mode="r")
ls = np.asarray(grid).max(axis=0)[::-1].astype(np.float32)
pam = np.asarray(raw).max(axis=1).astype(np.float32)


def to_grid(m, pitch_um):
    rows = np.arange(m.shape[0]) * pitch_um / 1000.0
    y = np.linspace(0, mm, N)
    out = np.empty((N, N), np.float32)
    for c in range(N):
        i = int(round(c / (N - 1) * (m.shape[1] - 1)))
        out[:, c] = np.interp(y, rows, m[:, i])
    return out


ls_g, pam_g = to_grid(ls, C.LS_PITCH_UM), to_grid(pam, C.PAM_FRAME_PITCH_UM)


def hp(m, s=15):
    return m - gaussian_filter(m, s)


hl, hpp = hp(ls_g), hp(pam_g)
hl = (hl - hl.mean()) / (hl.std() + 1e-9)
hpp = (hpp - hpp.mean()) / (hpp.std() + 1e-9)


def warp(s, rot, dx, dy):
    th = np.deg2rad(-rot)
    c, si = np.cos(th), np.sin(th)
    M = np.array([[s * c, -s * si], [s * si, s * c]])
    cx = cy = (N - 1) / 2
    off = np.array([cx, cy]) - M @ np.array([cx - dx * N / mm, cy - dy * N / mm])
    return affine_transform(hl, M, offset=off, order=1, mode="constant", cval=0)


def corr(w):
    a = w.ravel(); b = hpp.ravel()
    m = (a != 0)
    a, b = a[m], b[m]
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-9))


print(f"identity:                corr {corr(hl):+.4f}")
print(f"user (1.215,-4.4,1.0,0.53): corr {corr(warp(1.215, -4.4, 1.0, 0.53)):+.4f}")

best = (-9, None)
for s in np.arange(0.90, 1.31, 0.02):
    for rot in [-6, -4.4, -3, 0, 3]:
        for dx in [0.0, 0.5, 1.0]:
            for dy in [0.0, 0.5]:
                c = corr(warp(s, rot, dx, dy))
                if c > best[0]:
                    best = (c, (round(float(s), 2), rot, dx, dy))
print(f"search best: corr {best[0]:+.4f} at {best[1]}")
