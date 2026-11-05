"""
AR-PAM txt 的读取、双向扫描纠偏、归一化与缓存构建（v2.1）。

数据事实（本模块 mirror_report() 在 data/1、data/4 上实测得到）：
  * 每个 txt = 1 个 B-scan：1000 A-lines x 500 RF samples（tab 分隔浮点，未削顶）
  * 文件按时间顺序写入（约 1.0~1.4 s/帧），编号即采集顺序
  * **双向往返光栅扫描**：偶数帧 X 正扫，奇数帧 X 反扫（A-line 轴镜像）。
        直接相关(相邻帧)            ~0.17
        翻转奇数帧 A-line 轴后相关   ~0.93
        亮结构重合(直接 / 翻转)      0.01~0.29 / 0.93~0.96
        翻转后残余位移               dz=0, dx=0（就是精确的 [::-1]）
    => 必须翻转奇数帧再堆叠；翻转后 1 帧 = 10 um，1000 帧 = 10 mm。

  （勘误：v2.0 曾把奇偶判成"两套对比通道"并丢弃一半帧。那是把镜像检验
    做在了 RF 轴上的错误结论。见 REVIEW_v2.md §1。）

约定：
  raw txt 读入后转置为 (RF sample, A-line) = (500, 1000)，
  包络 env 为 (500, 1000)，缓存为 (N, 457, 1000)。
"""

import glob
import json
import os
import re
import time

import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import zoom

import config as C


# ---------------------------------------------------------------------------
# 基础 IO
# ---------------------------------------------------------------------------
def list_frames(txt_dir):
    """按编号排序返回 txt 路径列表（容忍缺号，例如 mouse4 缺 998/999）。"""
    out = []
    for f in glob.glob(os.path.join(txt_dir, "*.txt")):
        m = re.fullmatch(r"(\d+)\.txt", os.path.basename(f))
        if m:
            out.append((int(m.group(1)), f))
    out.sort(key=lambda t: t[0])
    return [f for _, f in out]


def frame_index(path):
    return int(re.fullmatch(r"(\d+)\.txt", os.path.basename(path)).group(1))


def should_flip(path):
    """奇数帧（B-scan 反扫）需要沿 A-line 轴翻转。"""
    return bool(C.PAM_FLIP_ODD_X and (frame_index(path) % 2 == 1))


def read_rf(path, flip_x=None):
    """
    读一个 B-scan，返回 (RF sample, A-line) 的 float32，已去每列直流。
    flip_x=None 时按文件编号自动判断（奇数帧翻转）。
    """
    if flip_x is None:
        flip_x = should_flip(path)
    rf = np.loadtxt(path, dtype=np.float32).T          # (500, 1000) = (RF, A-line)
    if flip_x:
        rf = rf[:, ::-1]                               # 对应 raw_rf[::-1, :]
    rf -= rf.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(rf, dtype=np.float32)


def rf_to_env(rf):
    return np.abs(hilbert(rf, axis=0)).astype(np.float32)


def envelope(path, flip_x=None):
    return rf_to_env(read_rf(path, flip_x))


# ---------------------------------------------------------------------------
# 双向扫描诊断
# ---------------------------------------------------------------------------
def _corr(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / d) if d > 0 else 0.0


def _mask_overlap(a, b, q=99.7, rad=1):
    ma = a > np.percentile(a, q)
    mb = b > np.percentile(b, q)
    mbd = np.zeros_like(mb)
    for dz in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            mbd |= np.roll(np.roll(mb, dz, 0), dx, 1)
    return float((ma & mbd).sum() / max(ma.sum(), 1))


def _residual_shift(a, b):
    """翻转后两帧之间还剩多少位移（应接近 0）。"""
    from scipy.ndimage import gaussian_filter
    a = gaussian_filter(a, 2); b = gaussian_filter(b, 2)
    a = a - a.mean(); b = b - b.mean()
    F = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
    F /= (np.abs(F) + 1e-12)
    c = np.fft.irfft2(F, s=a.shape)
    i = np.unravel_index(np.argmax(c), c.shape)
    dz = i[0] if i[0] <= a.shape[0] // 2 else i[0] - a.shape[0]
    dx = i[1] if i[1] <= a.shape[1] // 2 else i[1] - a.shape[1]
    return int(dz), int(dx), float(c.max())


def mirror_report(txt_dir, n_positions=3):
    """
    量化"奇数帧是否需要 X 翻转"。为了不把方向混在一起，这里**始终拿偶数帧
    与紧邻的奇数帧比**：
      corr_direct  : corr(even_raw, odd_raw)      —— 不翻转
      corr_flipped : corr(even_raw, flip(odd))    —— 翻转奇数帧
      overlap_*    : p99.7 亮结构重合
      residual_*   : 翻转后残余位移（应为 0）
    """
    files = list_frames(txt_dir)
    if len(files) < 8:
        raise ValueError(f"{txt_dir} 帧数不足")

    evens = [i for i in range(len(files) - 1)
             if frame_index(files[i]) % 2 == 0 and frame_index(files[i + 1]) % 2 == 1]
    picks = np.linspace(0, len(evens) - 1, n_positions).astype(int)

    cd, cf, od, of, shifts = [], [], [], [], []
    for p in picks:
        e = evens[p]
        for k in (0, 2, 4):
            if e + k + 1 >= len(files):
                break
            a = envelope(files[e + k], flip_x=False)
            b_raw = envelope(files[e + k + 1], flip_x=False)
            b_flip = b_raw[:, ::-1]
            cd.append(_corr(a, b_raw))
            cf.append(_corr(a, b_flip))
            od.append(_mask_overlap(a, b_raw))
            of.append(_mask_overlap(a, b_flip))
            shifts.append(_residual_shift(a, b_flip))

    shifts = np.array(shifts)
    cd_m, cf_m = float(np.mean(cd)), float(np.mean(cf))
    od_m, of_m = float(np.mean(od)), float(np.mean(of))
    return {
        "txt_dir": os.path.abspath(txt_dir),
        "n_files": len(files),
        "corr_direct": cd_m,
        "corr_flipped": cf_m,
        "overlap_direct": od_m,
        "overlap_flipped": of_m,
        "residual_shift_dz": int(np.median(shifts[:, 0])),
        "residual_shift_dx": int(np.median(shifts[:, 1])),
        "flip_needed": bool((cf_m - cd_m > 0.15) or (of_m - od_m > 0.20)),
    }


# ---------------------------------------------------------------------------
# 深度窗口
# ---------------------------------------------------------------------------
def estimate_surface_depth(path, aline_lo=150, aline_hi=850, step=10, rf_start=20):
    """
    单帧的"脑表面前沿"深度（RF sample）：中央 A-line 的包络首次超过
    max(6*noise, 0.18*col_max) 的位置，取 5% 分位（= 最浅的那部分）。
    rf_start=20 用来跳过 RF 记录开头的触发/振铃伪影（0~12 sample 有一条亮线）。
    """
    e = envelope(path)                                  # (RF, A-line)，已纠偏
    noise = float(np.median(e))
    first = []
    for al in range(aline_lo, aline_hi, step):
        col = e[:, al]
        thr = max(6.0 * noise, 0.18 * float(col.max()))
        idx = int(np.argmax(col[rf_start:] > thr)) + rf_start
        if col[idx] > thr:
            first.append(idx)
    return float(np.percentile(first, 5)) if first else np.nan


def estimate_window(files, n=24):
    """
    返回 (start, end, shallow)：
      start/end 用 config 的固定窗口（跨鼠一致，物理深度含义才一致）；
      shallow 是实测最浅表面，用于 QC 判断表面是否贴到窗口上沿。
    """
    idx = np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)
    depths = [estimate_surface_depth(files[i]) for i in idx]
    depths = [d for d in depths if np.isfinite(d)]
    shallow = float(np.percentile(depths, 5)) if depths else float(C.PAM_Z_START)
    return C.PAM_Z_START, C.PAM_Z_END, shallow


# ---------------------------------------------------------------------------
# 归一化 + 缓存
# ---------------------------------------------------------------------------
def frame_to_grid(path, z_start=None, z_end=None, bg=None):
    """
    单帧 -> (457, 1000) float32，只减背景、不除 gain、不 clip 上限。
    返回 (grid, bg)。
    """
    z_start = C.PAM_Z_START if z_start is None else z_start
    z_end = C.PAM_Z_END if z_end is None else z_end
    env = envelope(path)
    if bg is None:
        bg = float(np.median(env))
    sig = np.maximum(env - bg, 0.0)
    seg = sig[z_start:z_end, :]
    g = zoom(seg, (C.TARGET_Z / float(seg.shape[0]), 1.0), order=3)
    return np.maximum(g, 0.0).astype(np.float32), bg


def calibrate(files, z_start=None, z_end=None, n=30):
    """整卷标定：每帧背景（记录，用于复现）+ 全局 gain。"""
    z_start = C.PAM_Z_START if z_start is None else z_start
    z_end = C.PAM_Z_END if z_end is None else z_end
    idx = np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)
    bgs, p999 = [], []
    for i in idx:
        env = envelope(files[i])
        bg = float(np.median(env))
        bgs.append(bg)
        p999.append(float(np.percentile(
            np.maximum(env - bg, 0.0)[z_start:z_end], C.PAM_GAIN_PERCENTILE)))
    gain = float(np.median(p999)) * C.GAIN_HEADROOM
    return {"bg_median": float(np.median(bgs)), "gain": max(gain, 1e-9),
            "n_sampled": len(idx)}


def build_cache(txt_dir, out_npy, z_start=None, z_end=None, force=False, verbose=True,
                parity="all"):
    """
    构建 v2 缓存：**全部帧**（奇数帧自动翻转），整卷统一 gain，不 clip 上限。

    parity 参数只保留给"万一真的是两套对比通道"的情况：
      "all"（默认，双向扫描）| "odd" | "even"（旧的错误分支，仅兼容用）
    """
    files = list_frames(txt_dir)
    if not files:
        raise FileNotFoundError(f"没有 txt: {txt_dir}")

    if parity in ("odd", "even"):
        want = 1 if parity == "odd" else 0
        sel = [f for f in files if frame_index(f) % 2 == want]
        flip_flags = [False] * len(sel)
        if verbose:
            print(f"[warn] parity={parity}：丢弃一半帧，仅在确认是两套对比通道时使用")
    else:
        sel = files
        flip_flags = [should_flip(f) for f in sel]

    z_start = C.PAM_Z_START if z_start is None else z_start
    z_end = C.PAM_Z_END if z_end is None else z_end
    shallow = estimate_window(sel)[2]

    meta_path = out_npy + ".json"
    if os.path.exists(out_npy) and os.path.exists(meta_path) and not force:
        if verbose:
            print(f"[skip] {out_npy} 已存在（force=True 可重建）")
        return out_npy, json.load(open(meta_path, encoding="utf-8"))

    cal = calibrate(sel, z_start, z_end)
    gain = cal["gain"]

    if verbose:
        print(f"[cache] {txt_dir} 帧数={len(sel)} 翻转奇数帧={sum(flip_flags)} "
              f"窗口={z_start}:{z_end} gain={gain:.5f}")

    cache = np.empty((len(sel), C.TARGET_Z, C.TARGET_X), dtype=np.float32)
    t0 = time.time()
    bgs = []
    for i, f in enumerate(sel):
        g, bg = frame_to_grid(f, z_start, z_end, bg=None)
        cache[i] = g / gain
        bgs.append(bg)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sel)}  ({time.time() - t0:.0f}s)")

    np.save(out_npy, cache)

    meta = {
        "version": "v2.1",
        "txt_dir": os.path.abspath(txt_dir),
        "parity_mode": parity,
        "flip_odd_x": bool(C.PAM_FLIP_ODD_X),
        "n_files_total": len(files),
        "n_frames": len(sel),
        "n_flipped": int(sum(flip_flags)),
        "frame_pitch_um": C.PAM_FRAME_PITCH_UM,
        "z_start": int(z_start),
        "z_end": int(z_end),
        "z_pitch_um": C.PAM_RF_PITCH_UM,
        "gain": float(gain),
        "bg_median": cal["bg_median"],
        "bg_per_frame": [float(b) for b in bgs],
        "surface_depth_min_sample": float(shallow) if np.isfinite(shallow) else None,
        "clip_upper": None,
        "array": {"shape": list(cache.shape), "dtype": "float32"},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_files": [os.path.basename(f) for f in sel],
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[cache] 已保存 {out_npy} {cache.shape}  ({time.time() - t0:.0f}s)")
    return out_npy, meta


def load_cache(path, mmap=True):
    arr = np.load(path, mmap_mode="r" if mmap else None)
    meta_path = path + ".json"
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    return arr, meta


# ---------------------------------------------------------------------------
# 削顶/饱和自检
# ---------------------------------------------------------------------------
def clipping_report(arr, hi=1.0):
    a = np.asarray(arr, dtype=np.float32)
    sat = a >= hi
    flat = 0
    for k in range(min(a.shape[0], 64)):
        for c in range(0, a.shape[2], 4):
            col = sat[k, :, c]
            if not col.any():
                continue
            runs = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
            starts = np.where(runs == 1)[0]
            ends = np.where(runs == -1)[0]
            if len(starts) and (ends - starts).max() >= 3:
                flat += 1
    return {"frac_ge_hi": float(sat.mean()), "n_flat_alines": int(flat),
            "max": float(a.max()), "p999": float(np.percentile(a, 99.9)), "hi": hi}


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else C.PAM_M4_TXT_DIR
    rep = mirror_report(d)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    files = list_frames(d)
    print("window:", estimate_window(files))
    print("calib:", calibrate(files))
