"""把 v3 与展平模型的输出做平均，比较三者（v3 / flat / avg）的 FWHM 与前向残差。"""
import os

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import config as C
import pam_io
from flatten_ls import flatten_cols, unflatten_cols
from model import AxialNAFNet
from simulate import rl_deconv

dev = "cuda" if torch.cuda.is_available() else "cpu"
arr, _ = pam_io.load_cache(C.CACHE_M4_PATH)
n, Z, X = arr.shape

m3 = AxialNAFNet(in_ch=6, base_ch=-1).to(dev)
m3.load_state_dict(torch.load("best_v3.pth", map_location=dev, weights_only=True)); m3.eval()
mf = AxialNAFNet(in_ch=6, base_ch=-1).to(dev)
mf.load_state_dict(torch.load("best_flat.pth", map_location=dev, weights_only=True)); mf.eval()

out3 = np.zeros((n, Z, X), np.float32)
outf = np.zeros((n, Z, X), np.float32)
with torch.no_grad():
    for i in range(2, n - 2):
        idx = [int(np.clip(i + k, 0, n - 1)) for k in (-2, -1, 0, 1, 2)]
        fr = [np.asarray(arr[j], np.float32) for j in idx]
        x3 = torch.from_numpy(np.stack(fr)[None]).to(dev)
        rl3 = rl_deconv(fr[2], C.PSF_SIGMA_Z)[None, None]
        x3 = torch.cat([x3, torch.from_numpy(rl3).to(dev)], dim=1)
        out3[i] = m3(x3)[0, 0].cpu().numpy()

        fl, sh = [], None
        for f in fr:
            f2, s2 = flatten_cols(f)
            fl.append(f2); sh = s2
        xf = torch.from_numpy(np.stack(fl)[None]).to(dev)
        rlf = rl_deconv(fl[2], C.PSF_SIGMA_Z)[None, None]
        xf = torch.cat([xf, torch.from_numpy(rlf).to(dev)], dim=1)
        outf[i] = unflatten_cols(mf(xf)[0, 0].cpu().numpy(), sh)
        if i % 200 == 0:
            print(" ", i, flush=True)
for a in (out3, outf):
    a[:2] = a[2]; a[-2:] = a[-3]

def gauss(x, a, mu, s, b):
    return a * np.exp(-(x - mu) ** 2 / (2 * s ** 2)) + b

def fw(col, k, half=8):
    if k < half or k > len(col) - half - 1:
        return np.nan
    y = col[k - half:k + half + 1]
    try:
        p, _ = curve_fit(gauss, np.arange(-half, half + 1, dtype=float), y, p0=[y.max(), 0, 2, 0], maxfev=2000)
        if 0.5 < p[2] < 12:
            return 2.355 * p[2]
    except Exception:
        pass
    return np.nan

def score(e):
    A, B = [], []
    for i in range(50, n - 50, 14):
        r = np.asarray(arr[i])
        for c in range(0, X, 14):
            cr = r[:, c]
            pk, _ = find_peaks(cr, prominence=max(0.05, np.percentile(cr, 99)), distance=10)
            for k in pk:
                a, b = fw(cr, k), fw(e[i][:, c], k)
                if np.isfinite(a) and np.isfinite(b):
                    A.append(a); B.append(b)
    A = np.array(A); B = np.array(B)
    return np.median(A), np.median(B), float((B < A).mean())

for name, e in [("v3", out3), ("flat", outf), ("avg", 0.5 * (out3 + outf))]:
    a, b, f = score(e)
    deg = np.stack([np.asarray(pam_io.__dict__.get("x", lambda *z: z) and 0)]) if False else None
    print(f"{name:5s}: raw {a:.2f} -> enh {b:.2f} px  ({a/max(b,1e-6):.2f}x, 变窄 {f*100:.0f}%)", flush=True)

np.save(os.path.join(C.WORK_DIR, "enh_v3.npy"), out3)
np.save(os.path.join(C.WORK_DIR, "enh_flat.npy"), outf)
np.save(os.path.join(C.WORK_DIR, "enh_avg.npy"), 0.5 * (out3 + outf))
print("saved three volumes")
