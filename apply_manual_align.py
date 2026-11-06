"""应用用户手动对齐的参数（dx,dy,scale,rot）到光片 MIP，验证重叠是否提升。"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

import config as C
import pam_io

DX, DY, S, ROT = 0.96, 0.55, 1.220, -5.2      # mm, mm, x, deg（用户手调）
N = 600                                        # 公共网格 600x600 = 10x10 mm
mm = 10.0

raw, _ = pam_io.load_cache(C.CACHE_M4_PATH)
grid = np.load(C.IMS_M4_NPY, mmap_mode="r")
ls = np.asarray(grid).max(axis=0)[::-1].astype(np.float32)   # 翻转
pam = np.asarray(raw).max(axis=1).astype(np.float32)


def to_grid(m, pitch_um):
    rows_mm = np.arange(m.shape[0]) * pitch_um / 1000.0
    y = np.linspace(0, mm, N)
    x = np.linspace(0, mm, N)
    out = np.empty((N, N), np.float32)
    for c in range(N):
        col = c / (N - 1) * mm
        i = int(round(col / mm * (m.shape[1] - 1)))
        out[:, c] = np.interp(y, rows_mm, m[:, i])
    return out


ls_g = to_grid(ls, C.LS_PITCH_UM)
pam_g = to_grid(pam, C.PAM_FRAME_PITCH_UM)

# 把用户参数（以 600px=10mm 计）转成矩阵：先平移，再绕中心缩放旋转
px_per_mm = N / mm
tx, ty = DX * px_per_mm, DY * px_per_mm
cx = cy = (N - 1) / 2
th = np.deg2rad(-ROT)          # scipy 的角度方向与 CSS 相反
c, s = np.cos(th), np.sin(th)
M = np.array([[S * c, -S * s], [S * s, S * c]])       # 缩放+旋转
offset = np.array([cx, cy]) - M @ np.array([cx - tx, cy - ty])
ls_w = affine_transform(ls_g, M, offset=offset, order=1, mode="constant", cval=0)

for name, img in [("原始", ls_g), ("对齐后", ls_w)]:
    a = img > 0.25 * np.percentile(img, 99)
    b = pam_g > 0.25 * np.percentile(pam_g, 99)
    print(f"{name}: 光片覆盖 {a.mean()*100:.1f}%  血管落在组织内 {100*(a & b).sum()/max(b.sum(),1):.1f}%")

fig, axs = plt.subplots(1, 2, figsize=(13, 6.6))
for ax, img, t in [(axs[0], ls_g, "LS (as is)"), (axs[1], ls_w, "LS after manual align")]:
    rgb = np.zeros((N, N, 3), np.float32)
    a = img > 0.25 * np.percentile(img, 99)
    b = pam_g > 0.25 * np.percentile(pam_g, 99)
    rgb[..., 1] = np.clip(a * 0.6 + img / (np.percentile(img, 99.5) + 1e-9) * 0.4, 0, 1)
    rgb[..., 0] = np.clip(b * 0.85 + pam_g / (np.percentile(pam_g, 99.5) + 1e-9) * 0.15, 0, 1)
    rgb[..., 2] = np.clip((a & b) * 0.5, 0, 1)
    ax.imshow(np.clip(rgb, 0, 1), extent=[0, mm, mm, 0], aspect="equal")
    ax.set_title(t, fontsize=11); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
fig.suptitle(f"manual align: dx={DX}mm dy={DY}mm scale={S} rot={ROT}deg", fontsize=12)
fig.tight_layout()
p = os.path.join(C.QC_DIR, "overlay_manual_align2.png")
fig.savefig(p, dpi=160)
print("saved", p)
