"""
光片读取器（v2）。

保留 v1 的三个正确修正：
  1. 背景用每卷中位数（实测 415~422），而不是 vmin=350；
  2. X 方向用区间均值（4766 -> 1000，10 um 分箱）而不是最近邻抽样；
  3. 列方向平场（去掉沿 Z 的照明条纹），条纹低于噪声时等价于 no-op。

v2 改动：
  * 去掉 np.clip(..., 1.0)。光片最亮像素 3216、p99.9 约 900，
    gain = p99.9*1.3 时仍有 0.1% 的像素 >1；旧的 clip 会把最亮的
    几条血管削成平顶——而它正是要学的"锐利标签"。
  * 所有标定量写进 JSON 缓存，训练时多 worker 不重复扫 20GB 的 npy。
"""

import json
import os

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from scipy.ndimage import gaussian_filter1d

import config as C


def _open_volume(path):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    if h5py is None:
        raise ImportError("读取 .ims 需要 h5py；请先 pip install h5py")
    return h5py.File(path, "r")["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]


def block_mean_x(img, out_w=C.TARGET_X):
    """4766 -> 1000 的抗混叠降采样（区间平均）。img 必须是浮点。"""
    h, w = img.shape
    if w == out_w:
        return img.astype(np.float32, copy=False)
    img = np.asarray(img, dtype=np.float32)
    edges = np.linspace(0, w, out_w + 1).astype(np.int64)
    sums = np.add.reduceat(img, edges[:-1], axis=1)
    counts = np.diff(edges).astype(np.float32)
    return (sums / counts[None, :]).astype(np.float32)


def compute_ls_stats(ims_path, n_samples=16, verbose=False):
    """
    整卷标定：bg（背景中位数）、col_profile（列平场）、gain（p99.9*1.3）。
    这些都按"卷"估计，训练/推理/跨鼠一致。
    """
    vol = _open_volume(ims_path)
    n_y = vol.shape[1]
    ys = np.linspace(0, n_y - 1, min(n_samples, n_y)).astype(int)

    bgs, gains, cols = [], [], []
    for y in ys:
        raw = np.asarray(vol[0:C.TARGET_Z, y, :], dtype=np.float32)
        bg = float(np.percentile(raw, C.LS_BG_PERCENTILE))
        sig = np.maximum(raw - bg, 0.0)
        col = np.percentile(sig, C.LS_COL_PERCENTILE, axis=0).astype(np.float32)
        sig = np.maximum(sig - col[None, :], 0.0)
        bgs.append(bg)
        cols.append(col)
        gains.append(float(np.percentile(sig, C.LS_GAIN_PERCENTILE)))

    bg = float(np.median(bgs))
    gain = float(np.median(gains)) * C.GAIN_HEADROOM
    if not np.isfinite(gain) or gain <= 1.0:
        gain = float(np.median(gains) + 500.0)

    col_profile = np.median(np.stack(cols, axis=0), axis=0)
    col_profile = gaussian_filter1d(col_profile, C.LS_COL_SMOOTH).astype(np.float32)

    stats = {"bg": bg, "gain": gain, "col_profile": col_profile, "n_samples": len(ys)}
    if verbose:
        print(f"[LS] {os.path.basename(ims_path)} bg={bg:.1f} gain={gain:.1f} "
              f"col_std={col_profile.std():.2f}")
    return stats


def load_or_compute_stats(ims_path, cache_dir=None):
    """把标定量缓存成 json（col_profile 太大，单独存 npy）。网格则直接读侧车 json。"""
    if ims_path.endswith("_ls_grid.npy"):
        meta_path = ims_path + ".json"
        meta = json.load(open(meta_path, encoding="utf-8"))
        return {"bg": float(meta["bg"]), "gain": float(meta["gain"]),
                "col_profile": np.zeros(C.TARGET_X, np.float32), "grid": True}
    cache_dir = cache_dir or C.WORK_DIR
    key = os.path.splitext(os.path.basename(ims_path))[0]
    jp = os.path.join(cache_dir, f"ls_stats_{key}.json")
    npyp = os.path.join(cache_dir, f"ls_colprofile_{key}.npy")
    if os.path.exists(jp) and os.path.exists(npyp):
        st = json.load(open(jp, encoding="utf-8"))
        st["col_profile"] = np.load(npyp).astype(np.float32)
        return st
    st = compute_ls_stats(ims_path)
    json.dump({"bg": st["bg"], "gain": st["gain"], "n_samples": st["n_samples"]},
              open(jp, "w", encoding="utf-8"), indent=2)
    np.save(npyp, st["col_profile"])
    return st


class LightsheetReader:
    def __init__(self, ims_path, stats=None):
        self.ims_path = ims_path
        self.grid = ims_path.endswith("_ls_grid.npy")
        self._vol = None
        self._stats = stats

    @property
    def vol(self):
        if self._vol is None:
            self._vol = _open_volume(self.ims_path)
        return self._vol

    @property
    def stats(self):
        if self._stats is None:
            self._stats = load_or_compute_stats(self.ims_path)
        return self._stats

    def get_xz(self, y_idx, x_start=0):
        """
        y_idx: 全局 Y（npy 行号 + LS_Y_START，与 v1 约定一致；可为浮点，取最近行）。
        返回 (457, 1000) float32。
          * 网格路径：已经是"减背景+列平场+分箱+/gain"的结果，直接用；
          * 原始 npy：现场做这三步，不 clip 上限。
        """
        y = int(round(float(y_idx))) - C.LS_Y_START
        y = int(np.clip(y, 0, self.vol.shape[1] - 1))

        if self.grid:
            return np.asarray(self.vol[:, y, :], dtype=np.float32)

        raw = np.asarray(
            self.vol[0:C.TARGET_Z, y, x_start:x_start + C.LS_X_PIX], dtype=np.float32
        )
        st = self.stats
        sig = np.maximum(raw - st["bg"], 0.0)
        n = sig.shape[1]
        sig = np.maximum(sig - st["col_profile"][x_start:x_start + n][None, :], 0.0)
        sig = block_mean_x(sig, C.TARGET_X)
        return (sig / st["gain"]).astype(np.float32)     # 不 clip

    def surface_depth(self, y_idx, x_start=0, thr_frac=0.08):
        """
        粗略的脑表面深度（像素，0=窗口顶部），用于可选的"表面相对衰减"。
        返回 (457,1) 的逐列表面深度；失败时返回 0。
        """
        img = self.get_xz(y_idx, x_start)
        prof = gaussian_filter1d(img.max(axis=0), 3.0)
        thr = thr_frac * float(prof.max() + 1e-9)
        z = np.argmax(img > thr, axis=0).astype(np.float32)
        return gaussian_filter1d(z, 8.0)[None, :]
