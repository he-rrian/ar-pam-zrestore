"""导出两只小鼠的声光/增强 MIP（供与完整 .ims 光片 MIP 并排）。"""
import os

import numpy as np
from PIL import Image

import config as C
import pam_io

for m, cache in [("4", C.CACHE_M4_PATH), ("1", os.path.join(C.WORK_DIR, "1_pam_v2.npy"))]:
    raw, _ = pam_io.load_cache(cache)
    enh_p = os.path.join(C.WORK_DIR, f"mouse{m}_enhanced_v2.npy")
    if not os.path.exists(enh_p):
        print("missing", enh_p); continue
    enh = np.load(enh_p, mmap_mode="r")

    def mip(a):
        out = np.zeros((a.shape[0], a.shape[2]), np.float32)
        for i in range(a.shape[0]):
            out[i] = np.asarray(a[i]).max(axis=0)
        return out

    for name, arr in [("pam", raw), ("enh", enh)]:
        v = mip(arr)
        lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
        x = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.7
        img = (x * 255).astype(np.uint8)
        p = os.path.join(C.QC_DIR, f"mip_{name}_{m}.png")
        Image.fromarray(img).save(p)
        print("saved", p, v.shape)
