"""
零训练的物理反卷积基线：Wiener / Richardson-Lucy。

目的：在训练任何网络之前，先用标定好的 PSF 对真实 PAM 做一次反卷积。
如果 RL/Wiener 已经能把孤立血管的 Z 向 FWHM 明显收窄，而且不炸噪声，
那网络至少要比它更好，否则说明网络学到的是"高光滤镜"。

用法:
    python deconv_baseline.py --frame 200 --sigma 3.5
"""

import argparse
import glob
import os
import re

import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import zoom, gaussian_filter1d, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TARGET_Z, PSF_SIGMA_Z, PAM_Z_START, PAM_Z_END


def _files(txt_dir):
    files = glob.glob(os.path.join(txt_dir, "*.txt"))
    files = [f for f in files if re.search(r"\d+", os.path.basename(f))]
    return sorted(files, key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))


def load_frame(path):
    """v2：只减背景，不 clip 上限（和训练/缓存一致）。"""
    rf = np.loadtxt(path, dtype=np.float32).T
    rf = rf - rf.mean(axis=0, keepdims=True)
    env = np.abs(hilbert(rf, axis=0))[PAM_Z_START:PAM_Z_END]
    xz = zoom(env, (TARGET_Z / float(env.shape[0]), 1.0), order=3)
    xz = np.maximum(xz - np.median(xz), 0)
    gain = np.percentile(xz, 99.9) * 1.3 + 1e-9
    return (xz / gain).astype(np.float32)


def rl_deconv(img, sigma, iters=30):
    est = img.copy() + 1e-4
    for _ in range(iters):
        conv = gaussian_filter1d(est, sigma, axis=0, mode="nearest")
        est = est * gaussian_filter1d(img / (conv + 1e-6), sigma, axis=0, mode="nearest")
        est = np.clip(est, 0, None)
    return est


def wiener_deconv(img, sigma, snr=0.01):
    n = img.shape[0]
    f = np.fft.rfftfreq(n)
    h = np.exp(-2.0 * (np.pi * sigma * f) ** 2)
    g = h / (h ** 2 + snr)
    return np.maximum(np.fft.irfft(np.fft.rfft(img, axis=0) * g[:, None], n=n, axis=0), 0)


def fwhm(v):
    v = np.asarray(v, dtype=float)
    if v.max() <= 0:
        return np.nan
    p = int(np.argmax(v))
    half = v[p] / 2.0
    l = p
    while l > 0 and v[l] > half:
        l -= 1
    r = p
    while r < len(v) - 1 and v[r] > half:
        r += 1
    return float(r - l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", default="data/4")
    ap.add_argument("--frame", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=PSF_SIGMA_Z)
    ap.add_argument("--out", default="deconv_baseline.png")
    args = ap.parse_args()

    files = _files(args.txt_dir)
    raw = load_frame(files[args.frame])

    w = wiener_deconv(raw, args.sigma, snr=0.01)
    r = rl_deconv(raw, args.sigma, iters=25)

    # 找一个强而窄的孤立峰做 1D 对比
    col = int(np.argmax(raw.max(axis=0)))
    print(f"frame={args.frame}, profile column={col}")
    print(f"FWHM raw={fwhm(raw[:, col]):.1f}px  wiener={fwhm(w[:, col]):.1f}px  RL={fwhm(r[:, col]):.1f}px")

    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    for a, img, t in zip(ax[0], [raw, w, r],
                         ["raw PAM", f"Wiener sigma={args.sigma}", f"RL 25it sigma={args.sigma}"]):
        a.imshow(img, cmap="magma", aspect="auto", vmin=0, vmax=1)
        a.set_title(t)
        a.axis("off")
    ax[1, 0].plot(raw[:, col], label="raw")
    ax[1, 0].plot(w[:, col], label="wiener")
    ax[1, 0].plot(r[:, col], label="RL")
    ax[1, 0].legend()
    ax[1, 0].set_title("axial profile")
    ax[1, 1].plot(raw[:, col], label="raw")
    ax[1, 1].set_title("raw only")
    ax[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=170)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
