"""把 686 层新网格缩放到旧网格的标定口径，并用重叠的 457 层核实。"""
import os

import numpy as np

import config as C

NEW = os.path.join(C.WORK_DIR, "4_ls_grid_full.npy")
OLD = os.path.join(C.WORK_DIR, "4_ls_grid.npy")

old = np.load(OLD, mmap_mode="r")
new = np.load(NEW, mmap_mode="r")
n_ov = min(old.shape[0], new.shape[0])

# 重叠段抽样比较
a = np.asarray(old[::7, ::97, :]).astype(np.float32)
b = np.asarray(new[:n_ov][::7, ::97, :]).astype(np.float32)
k = float(a.mean() / (b.mean() + 1e-9))
print(f"overlap {n_ov} layers: old mean {a.mean():.4f}  new mean {b.mean():.4f}  k={k:.4f}")

# 分块缩放整卷
tmp = NEW + ".tmp.npy"
mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=new.shape)
for z0 in range(0, new.shape[0], 50):
    z1 = min(z0 + 50, new.shape[0])
    mm[z0:z1] = (np.asarray(new[z0:z1]).astype(np.float32) * k).astype(np.float16)
mm.flush(); del mm; del new; del old
os.replace(tmp, NEW)

chk = np.asarray(np.load(NEW, mmap_mode="r")[::7, ::97, :]).astype(np.float32)
print(f"after: mean {chk.mean():.4f} p99 {np.percentile(chk,99):.4f} (old p99 {np.percentile(a,99):.4f})")
