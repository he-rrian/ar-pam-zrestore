"""
把光片体数据压成"训练即用"的网格： (457, 4700, 1000) float16。

为什么需要：
  * 原始 npy 是 19 GB（X=4766），而训练每步只用到 block-mean 后的 1000 列；
    每次现场做 4766->1000 的区间均值，I/O 是 4.7 倍、CPU 也白烧。
  * 压成 4.3 GB 后，上传/训练读取都省 4~5 倍。

网格内容 = 已经做完 减背景 -> 列平场 -> X 区间均值 -> /整卷 gain 的结果，
所以 GridReader 不需要再归一化。

用法:
    python ls_grid.py --mouse 4          # -> work/4_ls_grid.npy + .json
"""

import argparse
import json
import os
import time

import numpy as np

import config as C
from data_reader import _open_volume, block_mean_x, compute_ls_stats


def build(mouse, chunk=25):
    src_path = C.IMS_M4_RAW if mouse == "4" else C.IMS_M1_RAW
    out = os.path.join(C.WORK_DIR, f"{mouse}_ls_grid.npy")
    vol = _open_volume(src_path)
    assert vol.shape[0] >= C.TARGET_Z

    st = compute_ls_stats(src_path)
    bg, gain, col = st["bg"], st["gain"], st["col_profile"]
    print(f"[grid] {src_path} bg={bg:.1f} gain={gain:.1f} col_std={col.std():.2f}")

    shape = (C.TARGET_Z, C.LS_Y_ROWS, C.TARGET_X)
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=shape)
    # npy 已经是裁好的（Y 0..4699），.ims 才需要加 LS_Y_START
    y_off = 0 if vol.shape[1] == C.LS_Y_ROWS else C.LS_Y_START
    t0 = time.time()
    for y0 in range(0, C.LS_Y_ROWS, chunk):
        y1 = min(y0 + chunk, C.LS_Y_ROWS)
        raw = np.asarray(vol[0:C.TARGET_Z, y_off + y0:y_off + y1, :], dtype=np.float32)
        sig = np.maximum(raw - bg, 0.0)
        sig = np.maximum(sig - col[None, None, :], 0.0)
        bm = np.empty((C.TARGET_Z, y1 - y0, C.TARGET_X), dtype=np.float32)
        for k in range(y1 - y0):
            bm[:, k, :] = block_mean_x(sig[:, k, :], C.TARGET_X)
        mm[:, y0:y1, :] = (bm / gain).astype(np.float16)
        if (y0 // chunk) % 20 == 0:
            print(f"  {y1}/{C.LS_Y_ROWS}  ({time.time() - t0:.0f}s)", flush=True)
    mm.flush()
    del mm

    meta = {"version": "grid-v1", "source": os.path.abspath(src_path),
            "shape": list(shape), "dtype": "float16",
            "bg": float(bg), "gain": float(gain),
            "z_pitch_um": C.LS_PITCH_UM, "x_pitch_um": C.LS_X_BIN_UM,
            "y_start": C.LS_Y_START, "y_rows": C.LS_Y_ROWS,
            "note": "已归一化：无需再减背景/除 gain"}
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[grid] saved {out} {shape} ({os.path.getsize(out) / 2**30:.2f} GB, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", required=True, choices=["1", "4"])
    args = ap.parse_args()
    build(args.mouse)
