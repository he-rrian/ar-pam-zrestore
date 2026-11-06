"""对若干 checkpoint 用真实数据快速筛 FWHM（只推理子集，快）。"""
import glob
import os
import sys

import numpy as np
import torch

import config as C
import pam_io
from model import AxialNAFNet
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from simulate import rl_deconv

dev = "cuda" if torch.cuda.is_available() else "cpu"


def gauss(x, a, mu, s, b):
    return a * np.exp(-(x - mu) ** 2 / (2 * s ** 2)) + b


def fw(col, k, half=8):
    if k < half or k > len(col) - half - 1:
        return np.nan
    y = col[k - half:k + half + 1]
    try:
        p, _ = curve_fit(gauss, x_ := np.arange(-half, half + 1, dtype=float), y,
                         p0=[y.max(), 0, 2, 0], maxfev=2000)
        if 0.5 < p[2] < 12:
            return 2.355 * p[2]
    except Exception:
        pass
    return np.nan


arr, _ = pam_io.load_cache(C.CACHE_M4_PATH)
idx = np.linspace(50, len(arr) - 50, 40).astype(int)


def infer(ckpt):
    m = AxialNAFNet(in_ch=6, base_ch=-1).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    m.eval()
    out = {}
    with torch.no_grad():
        for i in idx:
            ch = [np.asarray(arr[int(np.clip(i + k, 0, len(arr) - 1))], np.float32) for k in (-2, -1, 0, 1, 2)]
            x = torch.from_numpy(np.stack(ch)[None]).to(dev)
            rl = rl_deconv(ch[2], C.PSF_SIGMA_Z)[None, None]
            x = torch.cat([x, torch.from_numpy(rl).to(dev)], dim=1)
            out[i] = m(x)[0, 0].cpu().numpy()
    return out


def score(outs):
    fw_all, narrowed, n = [], 0, 0
    for i, e in outs.items():
        r = np.asarray(arr[i], np.float32)
        for c in range(0, e.shape[1], 12):
            col_r = r[:, c]
            pk, _ = find_peaks(col_r, prominence=max(0.05, np.percentile(col_r, 99)), distance=10)
            for k in pk:
                a, b = fw(col_r, k), fw(e[:, c], k)
                if np.isfinite(a) and np.isfinite(b):
                    fw_all.append((a, b))
                    narrowed += int(b < a)
                    n += 1
    if not n:
        return np.nan, 0.0, n
    a = np.array([x[0] for x in fw_all]); b = np.array([x[1] for x in fw_all])
    return float(np.median(a)), float(np.median(b)), float(narrowed / n)


for ck in sys.argv[1:]:
    if not os.path.exists(ck):
        continue
    a, b, frac = score(infer(ck))
    print(f"{ck:22s} raw {a:.2f} -> enh {b:.2f} px  ({a / max(b, 1e-6):.2f}x, narrowed {frac*100:.0f}%)", flush=True)
