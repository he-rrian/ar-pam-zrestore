import numpy as np
import config as C, pam_io

raw, _ = pam_io.load_cache(C.CACHE_M4_PATH)
grid = np.load(C.IMS_M4_NPY, mmap_mode="r")

def width_profile(mip, thr_frac=0.15):
    t = np.percentile(mip, 95) * thr_frac
    return (mip > t).sum(axis=1).astype(np.float32)

pam_mip = np.asarray(raw).max(axis=1)          # (998, 1000)
ls_mip = np.asarray(grid).max(axis=0)          # (4700, 1000)
step = 4700 // 998
ls_ds = ls_mip[: step * 998].reshape(998, step, -1).max(axis=1)

wp = width_profile(pam_mip); wl = width_profile(ls_ds)
def corr(a,b):
    a=(a-a.mean())/(a.std()+1e-9); b=(b-b.mean())/(b.std()+1e-9); return float((a*b).mean())
print(f"脑宽曲线: 同向 {corr(wl,wp):+.3f}  翻转 {corr(wl[::-1],wp):+.3f}")
# 亮度重心沿 Y 的位置（形状的上下不对称性）
def centroid(p):
    p = p - p.min(); return float((np.arange(len(p))*p).sum()/(p.sum()+1e-9))
print(f"亮度重心(行): LS {centroid(wl):.0f}  PAM {centroid(wp):.0f}  (总行数 998)")
