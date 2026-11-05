"""
光片 <-> 声光 的"相对脑表面深度"工具（§8 深度分层评估的基础）。

两种模态都要把深度轴换成"距离软脑膜（pia）多少 μm"，否则：
  * 光片的表面随 Y 起伏 ~300 层（630 μm），固定 457 层窗口在有些 Y 只剩 330 μm 皮层；
  * 声光的表面回波位置也随 Y 变（mouse4 在 RF 88~190）。

提供：
  pia_from_ls(img)      光片单层 -> 逐列表面 z（层）
  surface_from_pam(img) 声光单帧 -> 逐列表面 z（层）
  flatten(img, surf)    按表面展平到"表面下 k 层"
  valid_mask(...)       哪些 Y 的表面完整落在窗口内（方式 A：轻量筛选）
"""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter

import config as C


def _first_crossing(prof, thr):
    """一维剖面首次超过 thr 的下标；不到阈值返回 NaN。"""
    idx = np.flatnonzero(prof > thr)
    return float(idx[0]) if len(idx) else np.nan


def _smooth_surface(z, x_smooth=31):
    """沿 X 做中值 + 高斯平滑，去掉血管造成的局部跳变。"""
    z = np.asarray(z, dtype=np.float32)
    ok = np.isfinite(z)
    if ok.sum() < max(10, z.size // 10):
        return np.full_like(z, np.nan)
    if not ok.all():                      # 用有效列插值补齐
        idx = np.arange(z.size)
        z = np.interp(idx, idx[ok], z[ok]).astype(np.float32)
    z = median_filter(z, size=min(x_smooth, z.size // 3 * 2 + 1), mode="nearest")
    z = gaussian_filter1d(z, x_smooth / 4.0, mode="nearest")
    return z.astype(np.float32)


def pia_from_ls(img, thr_frac=0.15, smooth=31):
    """
    光片单层 (Z, X) -> 逐列表面 z（浮点，单位=层）。
    阈值取"背景 + thr_frac*(列最大-背景)"，对未归一化的原始 uint16 也成立。
    """
    a = gaussian_filter(np.asarray(img, dtype=np.float32), (2.0, 1.5))
    bg = float(np.median(a))
    z = np.full(a.shape[1], np.nan, dtype=np.float32)
    for x in range(a.shape[1]):
        col = a[:, x]
        mx = float(col.max())
        if mx - bg <= 1e-6:
            continue
        z[x] = _first_crossing(col, bg + thr_frac * (mx - bg))
    return _smooth_surface(z, smooth)


def surface_from_pam(img, thr_frac=0.15, smooth=31):
    """声光单帧 (Z, X)（包络）-> 逐列表面 z；同样是背景相对阈值。"""
    a = gaussian_filter(np.asarray(img, dtype=np.float32), (2.0, 1.5))
    bg = float(np.median(a))
    z = np.full(a.shape[1], np.nan, dtype=np.float32)
    for x in range(a.shape[1]):
        col = a[:, x]
        mx = float(col.max())
        if mx - bg <= 1e-6:
            continue
        z[x] = _first_crossing(col, bg + thr_frac * (mx - bg))
    return _smooth_surface(z, smooth)


def flatten(img, surf, out_z=320, dz=1.0):
    """
    按表面展平：输出第 k 行 = 表面下 k*dz 层。
    表面未知/越界的列输出 0（用 valid_columns 掩掉）。
    """
    a = np.asarray(img, dtype=np.float32)
    Z, X = a.shape
    out = np.zeros((out_z, X), dtype=np.float32)
    zz = np.arange(Z, dtype=np.float32)
    kk = np.arange(out_z, dtype=np.float32) * dz
    for x in range(X):
        s = surf[x]
        if not np.isfinite(s):
            continue
        src = s + kk
        ok = (src >= 0) & (src <= Z - 1)
        if ok.sum() < out_z // 2:
            continue
        out[ok, x] = np.interp(src[ok], zz, a[:, x])
    return out


def valid_columns(surf, Z, margin=None, need=320, colmax=None, amp_frac=0.15):
    """
    表面完整落在窗口内（上下都留够 need 层）的列。
    colmax 给定时，还要求该列有足够信号（排除空切片/噪声列）。
    """
    margin = C.LS_SURFACE_MARGIN if margin is None else margin
    ok = np.isfinite(surf) & (surf >= margin) & (surf <= Z - need)
    if colmax is not None:
        thr = amp_frac * float(np.percentile(colmax, 95))
        ok = ok & (np.asarray(colmax) > thr)
    return ok


def valid_y(reader, ys, margin=None, need=320):
    """方式 A：逐 Y 判断表面是否完整落在 457 层窗口内。"""
    out = {}
    for y in ys:
        img = reader.get_xz(y + C.LS_Y_START, 0)
        s = pia_from_ls(img)
        m = valid_columns(s, img.shape[0], margin, need)
        out[int(y)] = float(m.mean())
    return out


def depth_curve(flat, vessel_q=99.0):
    """
    展平后的体 -> 两条随"表面下深度"变化的曲线：
      density : 每层高于阈值的像素占比（血管密度）
      mean    : 每层平均强度
    """
    flat = np.asarray(flat, dtype=np.float32)
    thr = np.percentile(flat, vessel_q)
    dens = (flat > thr).mean(axis=1)
    mean = flat.mean(axis=1)
    return {"density": dens.astype(np.float32), "mean": mean.astype(np.float32),
            "thr": float(thr)}


def resample_curve(curve, n_out):
    return np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(curve)), curve)


class ImsReader:
    """
    直接从 .ims 读某个 Y 的整段 Z（不落盘）。

    分层评估需要"表面下 960 μm"，而 457 层的 npy 在表面起伏 300 层时不够用；
    与其再存一个 ~40 GB 的宽 slab，不如评估时按需读。需要 h5py。
    """

    def __init__(self, path, x_width=C.LS_X_PIX):
        import h5py
        self._f = h5py.File(path, "r")
        self._ds = self._f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
        self.Z, self.Y, self.X = self._ds.shape
        self.x0 = (self.X - x_width) // 2
        self.width = x_width

    def slice_xz(self, y_global, z0=0, z1=None, flip_z=False, bin_x=True):
        z1 = self.Z if z1 is None else z1
        blk = self._ds[z0:z1, int(y_global), self.x0:self.x0 + self.width].astype(np.float32)
        if flip_z:
            blk = blk[::-1]
        if bin_x and self.width != C.TARGET_X:
            edges = np.linspace(0, self.width, C.TARGET_X + 1).astype(np.int64)
            sums = np.add.reduceat(blk, edges[:-1], axis=1)
            blk = sums / np.diff(edges).astype(np.float32)[None, :]
        return blk

    def close(self):
        self._f.close()
