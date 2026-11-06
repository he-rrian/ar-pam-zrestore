"""直接从 .ims 生成完整视野 XY MIP（全 X、全 Y，Z 每 4 层取一次）。"""
import os
import sys

import h5py
import numpy as np
from PIL import Image

MOUSE = sys.argv[1] if len(sys.argv) > 1 else "4"
SRC = rf"D:\tmp\data\{MOUSE}.ims"
OUT = rf"D:\tmp\work\mip_full_{MOUSE}.png"
ZSTEP = 4

f = h5py.File(SRC, "r")
ds = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
NZ, NY, NX = ds.shape
print(f"{MOUSE}.ims shape {ds.shape}, z-step {ZSTEP}")

mip = np.zeros((NY, NX), np.float32)
for z0 in range(0, NZ, ZSTEP):
    blk = ds[z0, :, :].astype(np.float32)
    np.maximum(mip, blk, out=mip)
    if z0 % 200 == 0:
        print(f"  z {z0}/{NZ}", flush=True)

bg = np.percentile(mip, 5)
v = np.clip((mip - bg) / max(np.percentile(mip, 99.5) - bg, 1e-9), 0, 1) ** 0.7
img = (v * 255).astype(np.uint8)
Image.fromarray(img).resize((img.shape[1] // 3, img.shape[0] // 3), Image.LANCZOS).save(OUT)
print("saved", OUT, mip.shape)
