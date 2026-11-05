"""
真实 PAM 上的量化评估（不需要光片配准）。

指标：
  1. 孤立血管的 Z 向 FWHM：raw vs enhanced，看是否真的变窄；
  2. 空白区噪声/幻影：raw 无信号处的 enhanced 均值；
  3. 前向一致性：把 enhanced 用标定 PSF 重新模糊，能否还原 raw（允许一个全局增益）；
  4. 能量比 / 饱和比例。

用法:
    python evaluate_real.py --raw work/4_pam_v2.npy --enh work/mouse4_enhanced_v2.npy
"""

import argparse

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from config import TARGET_Z, TARGET_X, PSF_SIGMA_Z, ATT_EXP


def _gauss(x, a, mu, s, b):
    return a * np.exp(-(x - mu) ** 2 / (2.0 * s ** 2)) + b


def _fit_fwhm(profile, k, half=8):
    if k < half or k > len(profile) - half - 1:
        return np.nan
    y = profile[k - half:k + half + 1]
    if y.max() <= 1e-6:
        return np.nan
    x = np.arange(-half, half + 1, dtype=float)
    try:
        popt, _ = curve_fit(_gauss, x, y, p0=[y.max(), 0.0, 2.0, 0.0], maxfev=2000)
        a, mu, s, b = popt
        if 0.5 < s < 12 and abs(mu) < 4:
            return float(2.355 * s)
    except Exception:
        pass
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="work/4_pam_v2.npy")
    ap.add_argument("--enh", default="work/mouse4_enhanced_v2.npy")
    ap.add_argument("--n_frames", type=int, default=80)
    ap.add_argument("--sigma", type=float, default=PSF_SIGMA_Z)
    args = ap.parse_args()

    raw = np.load(args.raw, mmap_mode="r")
    enh = np.load(args.enh, mmap_mode="r")
    n = min(len(raw), len(enh))
    idx = np.linspace(0, n - 1, min(args.n_frames, n)).astype(int)

    fw_raw, fw_enh, fw_z = [], [], []
    bg_std, ghost_vals, energy = [], [], []
    fwd_res = []

    z = np.linspace(0, 1, raw.shape[1])[:, None]
    att = np.exp(-ATT_EXP * z)

    for i in idx:
        r = np.asarray(raw[i], dtype=np.float32)
        e = np.asarray(enh[i], dtype=np.float32)

        # 1. FWHM
        for c in range(0, r.shape[1], 4):
            col = r[:, c]
            pk, _ = find_peaks(col, prominence=max(0.05, np.percentile(col, 99.0)), distance=10)
            for k in pk:
                fr = _fit_fwhm(col, k)
                fe = _fit_fwhm(e[:, c], k)
                if np.isfinite(fr) and np.isfinite(fe) and fr < 25 and fe < 25:
                    fw_raw.append(fr)
                    fw_enh.append(fe)
                    fw_z.append(k)

        # 2. 空白区噪声/幻影
        empty = r < 0.02
        if empty.sum() > 1000:
            bg_std.append(float(e[empty].std()))
            ghost_vals.append(float(np.mean(np.maximum(e[empty] - 0.02, 0))))

        # 3. 前向一致性（允许一个全局增益 alpha）
        deg = gaussian_filter1d(e, args.sigma, axis=0) * att
        alpha = float((deg * r).sum() / ((deg * deg).sum() + 1e-8))
        fwd_res.append(float(np.linalg.norm(alpha * deg - r) / (np.linalg.norm(r) + 1e-8)))
        energy.append(float(e.mean() / (r.mean() + 1e-8)))

    fw_raw = np.array(fw_raw)
    fw_enh = np.array(fw_enh)

    print("=" * 70)
    print(f"frames: {len(idx)}   peaks: {len(fw_raw)}")
    if len(fw_raw):
        print(f"FWHM raw  median {np.median(fw_raw):.2f} px  (IQR {np.percentile(fw_raw, 25):.2f}-{np.percentile(fw_raw, 75):.2f})")
        print(f"FWHM enh  median {np.median(fw_enh):.2f} px  (IQR {np.percentile(fw_enh, 25):.2f}-{np.percentile(fw_enh, 75):.2f})")
        print(f"FWHM 收窄比例 median {np.median(fw_raw) / (np.median(fw_enh) + 1e-9):.2f}x")
        print(f"  真正变窄的峰占比 (enh < raw): {np.mean(fw_enh < fw_raw) * 100:.1f}%")
    # 轴向深度稳定性：分深度带看收窄倍数（深层不应比浅层明显变差）
    if len(fw_z):
        zz = np.array(fw_z); rr = np.array(fw_raw); ee = np.array(fw_enh)
        edges = [(0, 100), (100, 200), (200, 320), (320, raw.shape[1])]
        print("按深度的收窄倍数（Z 稳定性）:")
        for a, b in edges:
            m = (zz >= a) & (zz < b)
            if m.sum() >= 10:
                print(f"  z {a:>3}-{b:<3}: n={m.sum():4d}  raw {np.median(rr[m]):5.2f} -> enh {np.median(ee[m]):5.2f} px"
                      f"  ({np.median(rr[m]) / max(np.median(ee[m]), 1e-6):.2f}x)")

    print(f"空白区 enhanced std: {np.median(bg_std) if bg_std else float('nan'):.4f}")
    print(f"空白区 ghost 均值:   {np.median(ghost_vals) if ghost_vals else float('nan'):.4f}")
    print(f"能量比 (enh/raw):    {np.median(energy):.2f}x")
    print(f"前向一致性残差:      {np.median(fwd_res):.3f}  (越小越物理)")
    print("=" * 70)
    print("判读：")
    print("  * FWHM 收窄 1.3~2.5x 且 ghost 很小 -> 真去卷积；")
    print("  * FWHM 没变或反而变宽、ghost 大 -> 只是对比度/高光；")
    print("  * 前向残差 > 0.6 -> 输出与原始声光不自洽，存在脑补。")


if __name__ == "__main__":
    main()
