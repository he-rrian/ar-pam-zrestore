"""
从真实 AR-PAM RF 数据标定轴向 PSF 与深度衰减。

你在上一轮被告知 "sigma=7.6, kernel=31"，但这个值并没有从你的数据里量过。
这个脚本用孤立血管峰的包络做高斯拟合，给出真实 sigma 的分布；同时量出
真实 PAM 的深度衰减曲线。先跑它，再改 dataset/model 里的参数。

用法:
    python calibrate_psf.py --txt_dir data/4 --out psf_mouse4.json
    python calibrate_psf.py --txt_dir data/1 --out psf_mouse1.json
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.signal import hilbert, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import zoom

from config import TARGET_Z, PAM_Z_START, PAM_Z_END


def _gauss(x, a, mu, s, b):
    return a * np.exp(-(x - mu) ** 2 / (2.0 * s ** 2)) + b


def _files(txt_dir):
    files = glob.glob(os.path.join(txt_dir, "*.txt"))
    files = [f for f in files if re.search(r"\d+", os.path.basename(f))]
    return sorted(files, key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", default="data/4")
    ap.add_argument("--out", default="psf_calibration.json")
    ap.add_argument("--n_frames", type=int, default=40)
    ap.add_argument("--stride_x", type=int, default=4)
    args = ap.parse_args()

    files = _files(args.txt_dir)
    if not files:
        raise SystemExit(f"没有找到 txt: {args.txt_dir}")
    idx = np.linspace(0, len(files) - 1, min(args.n_frames, len(files))).astype(int)

    sigmas, fwhms = [], []
    profiles = []
    noise = []

    for fi in idx:
        rf = np.loadtxt(files[fi], dtype=np.float32).T
        rf = rf - rf.mean(axis=0, keepdims=True)
        env = np.abs(hilbert(rf, axis=0))[PAM_Z_START:PAM_Z_END]

        # 深度衰减曲线（先重采样到 457，便于直接对照 config 的 Z 网格）
        e457 = zoom(env, (TARGET_Z / float(env.shape[0]), 1.0), order=3)
        p = e457.mean(axis=1)
        profiles.append(p / (p.max() + 1e-9))
        noise.append(float(np.percentile(e457, 10)))

        # 孤立峰的高斯拟合
        thr = np.percentile(env, 99.0)
        for c in range(0, env.shape[1], args.stride_x):
            col = env[:, c]
            pk, _ = find_peaks(col, prominence=thr, distance=8)
            for k in pk:
                if k < 8 or k > len(col) - 9:
                    continue
                y = col[k - 8:k + 9]
                x = np.arange(-8, 9, dtype=float)
                try:
                    popt, _ = curve_fit(
                        _gauss, x, y,
                        p0=[col[k], 0.0, 2.0, 0.0],
                        maxfev=2000,
                    )
                    a, mu, s, b = popt
                    if 0.8 < s < 6.0 and a > 2.0 * thr and abs(mu) < 3.0:
                        sigmas.append(float(s))
                        fwhms.append(float(2.355 * s))
                except Exception:
                    pass

    sigmas = np.array(sigmas)
    fwhms = np.array(fwhms)
    profiles = np.array(profiles)
    prof = np.median(profiles, axis=0)

    if len(sigmas) == 0:
        raise SystemExit("没有拟合到足够的孤立峰；请检查 txt_dir 或调低阈值。")

    scale = TARGET_Z / float(PAM_Z_END - PAM_Z_START)
    # 从表面峰之后开始拟合指数衰减；不能从 z=0 开始（窗口起始在表面之前）
    pk = int(np.argmax(prof))
    z_fit = np.arange(min(pk + 50, len(prof) - 10), len(prof))
    m = prof[z_fit] > 0.01
    if m.sum() >= 5:
        coef = np.polyfit(z_fit[m], np.log(prof[z_fit][m]), 1)
        att_exp = float(np.clip(-coef[0] * len(prof), 0.0, 3.0))
    else:
        att_exp = 0.0

    out = {
        "txt_dir": os.path.abspath(args.txt_dir),
        "n_peaks": int(len(sigmas)),
        "sigma_samples_median": float(np.median(sigmas)),
        "sigma_samples_p10": float(np.percentile(sigmas, 10)),
        "sigma_samples_p90": float(np.percentile(sigmas, 90)),
        "sigma_px_median": float(np.median(sigmas) * scale),
        "sigma_px_p10": float(np.percentile(sigmas, 10) * scale),
        "sigma_px_p90": float(np.percentile(sigmas, 90) * scale),
        "fwhm_um_median": float(np.median(fwhms) * 3.0),
        "profile": [float(v) for v in prof],
        "attenuation_exp": att_exp,
        "noise_floor_p10": float(np.median(noise)),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=" * 64)
    print(f"文件: {args.txt_dir}")
    print(f"拟合孤立峰数: {len(sigmas)}")
    print(f"轴向 sigma: 中位数 {np.median(sigmas):.2f} samples = {np.median(sigmas) * scale:.2f} px")
    print(f"             p10 {np.percentile(sigmas, 10):.2f} samples = {np.percentile(sigmas, 10) * scale:.2f} px")
    print(f"             p90 {np.percentile(sigmas, 90):.2f} samples = {np.percentile(sigmas, 90) * scale:.2f} px")
    print(f"对应 FWHM: 中位数 {np.median(fwhms) * 3.0:.1f} um")
    print(f"深度衰减: A(z) ~= exp(-{att_exp:.2f} z), z 归一化到 [0,1]")
    print(f"背景噪声 p10: {np.median(noise):.5f}")
    print(f"已保存: {args.out}")
    print("=" * 64)
    print("把 sigma_px_p10 / sigma_px_median / sigma_px_p90 填进 config.py 的")
    print("PSF_SIGMA_Z_MIN / PSF_SIGMA_Z / PSF_SIGMA_Z_MAX，再把 attenuation_exp")
    print("填进 ATT_EXP。当前 config 里是 3.5 / [2.4, 5.2] / 1.3。")


if __name__ == "__main__":
    main()
