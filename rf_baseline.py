"""
零训练的物理基线（v2）：在 RF 域 vs 包络域做反卷积，量出"Z 到底能恢复多少"。

为什么要有这一步：
  网络学的是"逆退化"。如果它对孤立血管的 FWHM 收窄还不如一个 25 行的
  Wiener，那它学到的就只是对比度拉伸。先把及格线量出来，再决定网络结构。

四种处理：
  raw                 |hilbert(rf)|
  env-wiener          对包络做 Wiener（sigma_z，457 网格）
  env-RL              对包络做 Richardson-Lucy
  rf-wiener           对 RF 做 Wiener（sigma in RF samples），再取包络

用法:
    python rf_baseline.py --txt_dir data/4 --frames 100 200 300 --out qc/rf_baseline.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d, zoom

import config as C
import pam_io


def _grid(x):
    """(500, 1000) RF sample 网格 -> (457, 1000) 2.1 um 网格。"""
    seg = x[C.PAM_Z_START:C.PAM_Z_END]
    return zoom(seg, (C.TARGET_Z / float(seg.shape[0]), 1.0), order=3)


def wiener_1d(img, sigma, snr, axis=0):
    n = img.shape[axis]
    f = np.fft.rfftfreq(n)
    h = np.exp(-2.0 * (np.pi * sigma * f) ** 2)
    g = h / (h ** 2 + 1.0 / max(snr, 1e-9))
    F = np.fft.rfft(img, axis=axis)
    shape = [1] * img.ndim
    shape[axis] = len(f)
    return np.fft.irfft(F * g.reshape(shape), n=n, axis=axis)


def rl_1d(img, sigma, iters=25):
    est = np.maximum(img, 0) + 1e-4
    for _ in range(iters):
        conv = gaussian_filter1d(est, sigma, axis=0, mode="nearest")
        est = est * gaussian_filter1d(img / (conv + 1e-6), sigma, axis=0, mode="nearest")
        est = np.maximum(est, 0)
    return est


def fwhm_at(profile, k, half=10):
    """抛物线无关的 -6dB 全宽（线性插值），比高斯拟合稳健。"""
    if k < half or k > len(profile) - half - 1:
        return np.nan
    y = profile[k - half:k + half + 1].astype(float)
    if y.max() <= 0:
        return np.nan
    halfv = y.max() / 2.0
    i = int(np.argmax(y))
    l = i
    while l > 0 and y[l] > halfv:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r] > halfv:
        r += 1
    if l == 0 or r == len(y) - 1:
        return np.nan
    # 线性插值到半高
    fl = (y[l + 1] - halfv) / (y[l + 1] - y[l] + 1e-9)
    fr = (y[r - 1] - halfv) / (y[r - 1] - y[r] + 1e-9)
    return float((r - fr) - (l + fl))


def collect_fwhm(img, min_prom=0.15, max_n=400):
    out = []
    peaks_z = []
    for c in range(0, img.shape[1], 3):
        col = img[:, c]
        if col.max() <= 0:
            continue
        pk, props = find_peaks(col, prominence=min_prom * col.max(), distance=12)
        for k in pk:
            v = fwhm_at(col, k)
            if np.isfinite(v) and 2.0 < v < 40.0:
                out.append(v)
                peaks_z.append(k)
                if len(out) >= max_n:
                    return np.array(out), np.array(peaks_z)
    return np.array(out), np.array(peaks_z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", default=C.PAM_M4_TXT_DIR)
    ap.add_argument("--frames", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--snr", type=float, default=200.0)
    ap.add_argument("--out", default=os.path.join(C.QC_DIR, "rf_baseline.png"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    files = pam_io.list_frames(args.txt_dir)
    sigma_rf = C.PSF_SIGMA_Z * C.PAM_Z_WINDOW / float(C.TARGET_Z)   # px -> RF samples

    rows = []
    fig, axes = plt.subplots(len(args.frames), 4, figsize=(20, 4.2 * len(args.frames)),
                             squeeze=False)
    for r, fi in enumerate(args.frames):
        f = files[min(fi, len(files) - 1)]
        rf = pam_io.read_rf(f)
        raw = _grid(np.abs(hilbert(rf, axis=0)))

        env_wiener = _grid(np.abs(hilbert(wiener_1d(np.abs(hilbert(rf, axis=0)), sigma_rf * 1.0, args.snr, axis=0), axis=0)))
        env_rl = _grid(rl_1d(np.abs(hilbert(rf, axis=0)), sigma_rf, iters=25))
        rf_wiener = _grid(np.abs(hilbert(wiener_1d(rf, sigma_rf, args.snr, axis=0), axis=0)))

        # 统一增益（按 raw 的 p99.9），便于同尺度比较
        g = np.percentile(raw, 99.9) + 1e-9
        imgs = [raw / g, env_wiener / g, env_rl / g, rf_wiener / g]
        names = ["raw", f"env-Wiener s={sigma_rf:.2f}", f"env-RL s={sigma_rf:.2f}",
                 f"RF-Wiener s={sigma_rf:.2f}"]

        stat = {}
        for img, name in zip(imgs, names):
            fw, _ = collect_fwhm(img)
            stat[name] = (float(np.median(fw)) if len(fw) else np.nan, len(fw))
        rows.append((pam_io.frame_index(f), stat))

        for c, (img, name) in enumerate(zip(imgs, names)):
            axes[r, c].imshow(img, cmap="magma", aspect="auto", vmin=0,
                              vmax=float(np.percentile(imgs[0], 99.9)))
            axes[r, c].set_title(f"{name}\nmedian FWHM {stat[name][0]:.1f} px (n={stat[name][1]})")
            axes[r, c].axis("off")

    fig.suptitle(f"RF vs envelope deconvolution — sigma_rf={sigma_rf:.2f} samples, snr={args.snr}",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)

    print("=" * 78)
    print(f"axial PSF sigma: {C.PSF_SIGMA_Z:.2f} px(457) = {sigma_rf:.2f} RF samples")
    print(f"{'frame':>6} | " + " | ".join(f"{n:>22}" for n in
                                          ["raw", "env-Wiener", "env-RL", "RF-Wiener"]))
    for fi, stat in rows:
        vals = [stat[n][0] for n in ["raw", f"env-Wiener s={sigma_rf:.2f}",
                                     f"env-RL s={sigma_rf:.2f}", f"RF-Wiener s={sigma_rf:.2f}"]]
        print(f"{fi:>6} | " + " | ".join(f"{v:>22.1f}" for v in vals))

    base = np.nanmedian([s["raw"][0] for _, s in rows])
    for k in ["env-Wiener", "env-RL", "RF-Wiener"]:
        v = np.nanmedian([s[f"{k} s={sigma_rf:.2f}"][0] for _, s in rows])
        print(f"{k:>12}: {base:.1f} -> {v:.1f} px  ({base / max(v, 1e-9):.2f}x)")
    print("判读：这就是零训练物理方法的上限。网络若达不到这个收窄倍数，"
          "说明它没有真的在反卷积。")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
