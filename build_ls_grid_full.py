"""A 方案：构建与 PAM 20:500 严格对齐的光片网格 (686, 4700, 1000) float16。
ims z = 2047-686 .. 2047，翻转后第 0 层对应 PAM 第 20 个采样（老裁法的同一起点）。
"""
import json
import os
import time

import h5py
import numpy as np

import config as C
from data_reader import block_mean_x, compute_ls_stats

MOUSE = "4"
SRC = r"D:\tmp\data\4.ims"
OUT = os.path.join(C.WORK_DIR, f"{MOUSE}_ls_grid_full.npy")
NZ = 686
Z0, Z1 = 1311, 1311 + NZ     # 老裁法 npy z=0 对应 ims z=1996

st = compute_ls_stats(SRC)
bg, gain, col = st["bg"], st["gain"], st["col_profile"]
print(f"[grid-full] bg={bg:.1f} gain={gain:.1f} z={Z0}:{Z1} nz={NZ}")

f = h5py.File(SRC, "r")
ds = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
print("ims", ds.shape)
shape = (NZ, C.LS_Y_ROWS, C.TARGET_X)
mm = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float16, shape=shape)

t0 = time.time()
chunk = 25
for y0 in range(0, C.LS_Y_ROWS, chunk):
    y1 = min(y0 + chunk, C.LS_Y_ROWS)
    raw = ds[Z0:Z1, C.LS_Y_START + y0:C.LS_Y_START + y1, :].astype(np.float32)
    raw = raw[::-1]                                  # 翻转，与老裁法一致
    sig = np.maximum(raw - bg, 0.0)
    sig = np.maximum(sig - col[None, None, :], 0.0)
    bm = np.empty((NZ, y1 - y0, C.TARGET_X), dtype=np.float32)
    for k in range(y1 - y0):
        bm[:, k, :] = block_mean_x(sig[:, k, :], C.TARGET_X)
    mm[:, y0:y1, :] = (bm / gain).astype(np.float16)
    if (y0 // chunk) % 20 == 0:
        print(f"  {y1}/{C.LS_Y_ROWS}  ({time.time() - t0:.0f}s)", flush=True)
mm.flush(); del mm; f.close()

meta = {"version": "grid-full", "shape": list(shape), "dtype": "float16",
        "bg": float(bg), "gain": float(gain), "z_pitch_um": C.LS_PITCH_UM,
        "x_pitch_um": C.LS_X_BIN_UM, "y_start": C.LS_Y_START,
        "y_rows": C.LS_Y_ROWS, "ims_z": [Z0, Z1],
        "note": "已归一化；第 0 层与 PAM sample 20 对齐"}
json.dump(meta, open(OUT + ".json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("saved", OUT, shape, f"{os.path.getsize(OUT)/2**30:.2f} GB")
