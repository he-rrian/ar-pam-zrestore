"""
.ims -> .npy 提取（v2.1，裁剪方向已按 h5py 实测校正）。

为什么改成这样：
  原始 .ims 是 (Z=2048, Y=6272, X=5504/4864)，旧脚本固定裁 457 层：
    mouse4: vol[1540:1997][::-1]   -> 背侧表面（高 Z，约 1900~2000）距上沿只剩 0~50 层
    mouse1: vol[50:507]            -> 背侧表面（低 Z，约 100）距上沿 50 层
  实测映射（corr=1.000）：mouse4 npy z = 1996 - ims z；mouse1 npy z = ims z - 50。
  背侧在**高 Z** 的 mouse4，"向上留白"必须**增大** Z 索引：
    mouse4 -> vol[1590:2047][::-1]（多 51 层背景）
    mouse1 -> vol[0:457]          （多 50 层背景）

用法:
    python npy.py --mouse 4 --verify          # 只打印映射/统计，不写
    python npy.py --mouse 4                   # 重裁 457 层（覆盖 data/4_subset.npy）
    python npy.py --mouse 4 --tall --out data/4_tall.npy   # 700 层，分层评估用
"""

import argparse
import os

import numpy as np

import config as C

try:
    import h5py
except ImportError:
    h5py = None


def _open_ims(path):
    if h5py is None:
        raise ImportError("读取 .ims 需要 h5py：pip install h5py（或用有 h5py 的环境，如 conda env wk）")
    return h5py.File(path, "r")["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]


def extract(mouse, z_start, n_layers, flip_z, out_path, chunk=100):
    ims = os.path.join(C.DATA_DIR, f"{mouse}.ims")
    src = _open_ims(ims)
    Z, Y, X = src.shape
    if z_start < 0 or z_start + n_layers > Z:
        raise ValueError(f"Z 越界：{z_start}+{n_layers} > {Z}")

    x0 = (X - C.LS_X_PIX) // 2
    x1 = x0 + C.LS_X_PIX
    if x0 < 0:
        raise ValueError(f"X 宽度不足：{X} < {C.LS_X_PIX}")

    shape = (n_layers, C.LS_Y_ROWS, C.LS_X_PIX)
    print(f"[extract] {ims}  Z[{z_start}:{z_start + n_layers}] flip={flip_z}  "
          f"Y[{C.LS_Y_START}:{C.LS_Y_START + C.LS_Y_ROWS}]  X[{x0}:{x1}]  -> {shape}")

    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=shape)
    for y0 in range(0, C.LS_Y_ROWS, chunk):
        y1 = min(y0 + chunk, C.LS_Y_ROWS)
        blk = src[z_start:z_start + n_layers,
                  C.LS_Y_START + y0:C.LS_Y_START + y1, x0:x1]
        if flip_z:
            blk = blk[::-1, :, :]
        mm[:, y0:y1, :] = blk
        print(f"  {y1}/{C.LS_Y_ROWS}", flush=True)
    mm.flush()
    del mm
    print(f"[extract] saved {out_path} ({os.path.getsize(out_path) / 2**30:.2f} GB)")


def verify(mouse, z_start, n_layers, flip_z):
    """不写文件：核对映射，并估计背侧表面在裁剪窗口里的位置。"""
    ims = os.path.join(C.DATA_DIR, f"{mouse}.ims")
    src = _open_ims(ims)
    Z, Y, X = src.shape
    x0 = (X - C.LS_X_PIX) // 2
    x1 = x0 + C.LS_X_PIX
    print(f"[verify] {ims} shape={src.shape}  x0={x0}")
    print(f"[verify] 裁剪 Z[{z_start}:{z_start + n_layers}] flip={flip_z}; "
          f"上方剩余 {Z - (z_start + n_layers)} 层, 下方剩余 {z_start} 层")
    if z_start + n_layers > Z:
        print("  !! Z 越界")
        return

    # 背侧表面：沿 Z 取 p99 over X，找组织起始。
    # mouse4 背侧在高 Z（翻转后落在 npy 小 z）；mouse1 背侧在低 Z（不翻）。
    ys = np.arange(0, C.LS_Y_ROWS, 400)
    onsets = []
    for y in ys:
        blk = src[z_start:z_start + n_layers, C.LS_Y_START + y, x0:x1].astype(np.float32)
        prof = np.percentile(blk, 99, axis=1)
        med = float(np.median(prof))
        thr = med + 0.20 * (float(prof.max()) - med)
        idx = np.flatnonzero(prof > thr)
        if len(idx) == 0:
            continue
        i = int(idx[-1]) if flip_z else int(idx[0])      # 背侧所在的一端
        onsets.append((n_layers - 1 - i) if flip_z else i)
    if onsets:
        o = np.array(onsets)
        print(f"[verify] 表面在 npy 坐标：min {o.min()}  median {np.median(o):.0f}  max {o.max()}")
        print(f"[verify] 表面到窗口上沿的余量：最少 {o.min()} 层（{o.min() * C.LS_PITCH_UM:.0f} um）")
    print("[verify] 映射：npy z = %s" % (
        f"{z_start + n_layers - 1} - ims z" if flip_z else f"ims z - {z_start}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", required=True, choices=["1", "4"])
    ap.add_argument("--z_start", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--tall", action="store_true", help=f"用 {C.LS_CROP_TALL_N} 层的宽 slab")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    cfg = C.LS_CROP[args.mouse]
    z_start = args.z_start if args.z_start is not None else cfg["z_start"]
    n_layers = args.n_layers if args.n_layers is not None else cfg["n_layers"]
    flip_z = cfg["flip_z"]
    if args.tall:
        n_layers = args.n_layers if args.n_layers is not None else C.LS_CROP_TALL_N
        z_start = min(z_start, 2048 - n_layers)

    if args.verify:
        verify(args.mouse, z_start, n_layers, flip_z)
        return

    out = args.out or os.path.join(C.DATA_DIR, f"{args.mouse}_subset.npy")
    if os.path.exists(out):
        print(f"!! {out} 已存在。换 --out，或先手动移走（本脚本不覆盖）")
        return
    extract(args.mouse, z_start, n_layers, flip_z, out)


if __name__ == "__main__":
    main()
