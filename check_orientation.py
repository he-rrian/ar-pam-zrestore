"""判断光片网格相对声光是否 Y 翻转 / Z 翻转（用分布曲线相关，不做配准）。"""
import numpy as np

import config as C
import pam_io

raw, _ = pam_io.load_cache(C.CACHE_M4_PATH)
grid = np.load(C.IMS_M4_NPY, mmap_mode="r")     # (457, 4700, 1000)

# ---- Y 方向：两模态各自的"每行平均强度"曲线 ----
pam_y = np.asarray(raw).mean(axis=(1, 2))       # (998,)
ls_y = np.asarray(grid).mean(axis=(0, 2))       # (4700,)
n = min(len(pam_y), len(ls_y))
# 把光片 4700 行块平均到 998 行
step = len(ls_y) // len(pam_y)
ls_y_ds = ls_y[: step * len(pam_y)].reshape(len(pam_y), step).mean(axis=1)


def corr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())


print(f"Y 方向: corr(同向) = {corr(ls_y_ds, pam_y):+.3f}   corr(翻转) = {corr(ls_y_ds[::-1], pam_y):+.3f}")

# ---- Z 方向：每层的平均强度曲线 ----
pam_z = np.asarray(raw).mean(axis=(0, 2))       # (457,)
ls_z = np.asarray(grid).mean(axis=(1, 2))       # (457,)
print(f"Z 方向: corr(同向) = {corr(ls_z, pam_z):+.3f}   corr(翻转) = {corr(ls_z[::-1], pam_z):+.3f}")

# 表面位置作为更强证据
import surface as S
pia_ls = S.pia_from_ls(np.asarray(grid[:, 2350, :], np.float32))
pia_pam = S.pia_from_ls(np.asarray(raw[500], np.float32))
print(f"表面所在层: LS 中位 {np.nanmedian(pia_ls):.0f}   PAM 中位 {np.nanmedian(pia_pam):.0f}")
