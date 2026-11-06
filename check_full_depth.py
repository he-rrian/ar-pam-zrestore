"""全深度 demo 的 FWHM 检查：只看原来的训练段(0-320 采样=0-1020 um)是否保住。"""
import os

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import config as C

arr = np.load(os.path.join(C.WORK_DIR, "4_pam_full_demo.npy"))[2:10]
enh = np.load(os.path.join(C.WORK_DIR, "4_full_enhanced_demo.npy"))


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


for name, hi in [("训练段 0-1020um", 486), ("全段 0-1500um", 714)]:
    a_all, b_all = [], []
    for i in range(arr.shape[0]):
        r = arr[i][:hi]; e = enh[i][:hi]
        for c in range(0, 1000, 15):
            cr = r[:, c]
            pk, _ = find_peaks(cr, prominence=max(0.05, np.percentile(cr, 99)), distance=10)
            for k in pk:
                fa, fb = fw(cr, k), fw(e[:, c], k)
                if np.isfinite(fa) and np.isfinite(fb):
                    a_all.append(fa); b_all.append(fb)
    a = np.array(a_all); b = np.array(b_all)
    print(f"{name}: n={len(a)}  raw {np.median(a):.2f} -> enh {np.median(b):.2f} px "
          f"({np.median(a)/max(np.median(b),1e-6):.2f}x, 变窄 {100*(b<a).mean():.0f}%)")
