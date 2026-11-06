"""尺度修正：把声光 686px(1440um) 重采样到与光片相同的物理高度 (686 层 x 1.842um = 1264um)。"""
import os

import numpy as np

import config as C

SCALE = 1.14                      # 实测：声光间距 / 光片间距
LS_PITCH_TRUE = C.LS_PITCH_UM / SCALE      # 1.842 um
LS_HEIGHT = C.TARGET_Z * LS_PITCH_TRUE     # 1263.7 um
src_rows_used = LS_HEIGHT / (C.PAM_RF_PITCH_UM)   # 480 * ... in samples? 用 480 样本=1440um
fraction = LS_HEIGHT / (480 * C.PAM_RF_PITCH_UM)
print(f"LS 真实像素 {LS_PITCH_TRUE:.3f} um, 光片高度 {LS_HEIGHT:.1f} um, 占声光窗口 {fraction*100:.1f}%")

src = np.load(os.path.join(C.WORK_DIR, "4_pam_full686s.npy"), mmap_mode="r")
n = src.shape[0]
take = fraction * 685.0
idx = np.arange(686) * take / 685.0
i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, 685)
w = (idx - i0).astype(np.float32)[None, :, None]

out = np.empty((n, 686, src.shape[2]), np.float32)
for k in range(n):
    a = np.asarray(src[k])[i0]; b = np.asarray(src[k])[i1]
    out[k] = a * (1 - w) + b * w
np.save(os.path.join(C.WORK_DIR, "4_pam_full686_s114.npy"), out)
print("saved", out.shape, float(out.mean()))
