"""全深度推理：输入 714 px（1500 um）的 12 帧，输出同样高度，并与原始对比出图。"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from model import AxialNAFNet
from simulate import rl_deconv

dev = "cuda" if torch.cuda.is_available() else "cpu"
arr = np.load(os.path.join(C.WORK_DIR, "4_pam_full_demo.npy"))   # (12, 714, 1000)
print("demo cache", arr.shape)

model = AxialNAFNet(in_ch=6, base_ch=-1).to(dev)
model.load_state_dict(torch.load("best_v3.pth", map_location=dev, weights_only=True))
model.eval()

outs = {}
with torch.no_grad():
    for i in range(2, 10):
        ch = [arr[i + k] for k in (-2, -1, 0, 1, 2)]
        rl = rl_deconv(ch[2], C.PSF_SIGMA_Z)[None, None]
        x = torch.from_numpy(np.stack(ch)[None]).to(dev)
        x = torch.cat([x, torch.from_numpy(rl).to(dev)], dim=1)
        outs[i] = model(x)[0, 0].cpu().numpy()
np.save(os.path.join(C.WORK_DIR, "4_full_enhanced_demo.npy"), np.stack([outs[i] for i in sorted(outs)]))

frames = [3, 5, 7]
fig, axs = plt.subplots(len(frames), 2, figsize=(14, 3.2 * len(frames)))
for r, i in enumerate(frames):
    for c, (img, t) in enumerate([(arr[i], f"raw PAM (full 1500 um)"), (outs[i], "enhanced (full)")]):
        ax = axs[r, c]
        ax.imshow(img, cmap="magma", aspect="auto", extent=[0, 10, 1.5, 0], vmin=0,
                  vmax=float(np.percentile(img, 99.9)))
        ax.set_title(t, fontsize=10); ax.set_xlabel("X (mm)")
        if c == 0:
            ax.set_ylabel("Z (mm)")
fig.suptitle("Full-depth test: can the model cover the whole 1500 um txt?", fontsize=12)
fig.tight_layout()
out = os.path.join(C.QC_DIR, "full_depth_test.png")
fig.savefig(out, dpi=160)
print("saved", out, "enh mean", float(np.mean([outs[i].mean() for i in outs])))
