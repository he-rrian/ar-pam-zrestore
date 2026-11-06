"""标定横向 PSF：扫 σ_x，让"光片模糊后"的横向自相关宽度与真实声光匹配。"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import config as C
import pam_io

raw, _ = pam_io.load_cache(C.CACHE_M4_PATH)
grid = np.load(C.IMS_M4_NPY, mmap_mode="r")


def acf_halfwidth_x(m, maxlag=40):
    """沿 X 的平均自相关半宽（px）。先去掉沿 X 的慢变包络。"""
    a = np.asarray(m, np.float32)
    a = a - gaussian_filter(a, (0, 12))          # 高通，去包络
    a = a - a.mean(axis=1, keepdims=True)
    F = np.fft.rfft(a, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=a.shape[1], axis=1)
    ac = ac / (ac[:, :1] + 1e-12)
    prof = ac.mean(axis=0)
    for i in range(1, maxlag):
        if prof[i] < 0.5:
            return (i - 1) + (prof[i - 1] - 0.5) / (prof[i - 1] - prof[i] + 1e-9)
    return np.nan


# 真实声光（多帧平均）
pam_w = []
for fi in [200, 380, 560, 740, 860]:
    pam_w.append(acf_halfwidth_x(np.asarray(raw[fi])))
pam_w = float(np.median(pam_w))
print(f"真实声光 横向 ACF 半宽 = {pam_w:.2f} px = {pam_w*2.1:.0f} um")

# 光片切成同样网格（用 10um 的网格）
ls = np.asarray(grid[:, 2350, :], np.float32)
print(f"{'sigma_x(px)':>11} {'模糊后ACF':>10} {'与声光差':>9}")
best = None
for sx in np.arange(0.4, 2.01, 0.1):
    b = gaussian_filter1d(ls, sx, axis=1)
    w = acf_halfwidth_x(b)
    d = abs(w - pam_w)
    print(f"{sx:11.1f} {w:10.2f} {d:9.2f}")
    if best is None or d < best[1]:
        best = (sx, d, w)
print(f"\n最佳 σ_x = {best[0]:.2f} px = {best[0]*2.1:.1f} um "
      f"(sigma) -> FWHM {best[0]*2.355*2.1:.0f} um")
