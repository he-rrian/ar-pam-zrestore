"""两只小鼠的对比图：光片=完整 .ims MIP，声光=原始/增强 MIP（各自真实物理尺寸）。"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = r"D:\tmp\work"
OUT = r"D:\tmp\result\6\compare_full_mip.png"

# 真实物理尺寸 (mm)
LS_MM = (5504 * 2.1 / 1000, 6272 * 2.1 / 1000)      # X, Y  11.56 x 13.17
PAM_MM = {4: (10.0, 9.98), 1: (10.0, 10.0)}

fig = plt.figure(figsize=(19, 8.6))
gs = fig.add_gridspec(2, 3, width_ratios=[LS_MM[0], PAM_MM[4][0], PAM_MM[4][0]])
for r, m in enumerate([4, 1]):
    imgs = [("light-sheet (full .ims MIP)", os.path.join(W, f"mip_full_{m}.png"), LS_MM),
            ("raw PAM (full depth MIP)", os.path.join(W, f"mip_pam_{m}.png"), PAM_MM[m]),
            ("enhanced (full depth MIP)", os.path.join(W, f"mip_enh_{m}.png"), PAM_MM[m])]
    for c, (title, path, (wx, wy)) in enumerate(imgs):
        ax = fig.add_subplot(gs[r, c])
        a = plt.imread(path)
        if c == 0:
            a = a[::-1]                      # 按用户要求上下翻转
        ax.imshow(a, cmap="gray", extent=[0, wx, wy, 0], aspect="equal")
        ax.set_title(f"Mouse{m}: {title}\n{wx:.2f} x {wy:.2f} mm", fontsize=10)
        ax.set_xlabel("X (mm)", fontsize=8)
        if c == 0:
            ax.set_ylabel("Y (mm)", fontsize=8)
fig.suptitle("Full-field comparison (light-sheet from complete .ims, no crop; PAM at its own 10 mm)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
