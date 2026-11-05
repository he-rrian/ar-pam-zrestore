"""
推理 v2：对拆好奇偶、按 v2 归一化的真实声光做 2.5D 反卷积。

    python infer.py --ckpt best_v2.pth --mouse all

输出：
    data/{1,4}_pam_v2.npy           原始（未增强）缓存
    data/mouse{1,4}_enhanced_v2.npy 增强结果（同一网格）
    qc/result_Mouse{1,4}_*.png      原始/增强对比
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io
from model import AxialNAFNet


def ensure_cache(mouse, force=False):
    d = C.PAM_M4_TXT_DIR if mouse == "4" else C.PAM_M1_TXT_DIR
    out = C.CACHE_M4_PATH if mouse == "4" else C.CACHE_M1_PATH
    if not os.path.exists(out) or force:
        pam_io.build_cache(d, out, force=force)
    else:
        print(f"[cache] 复用 {out}")
    return out


def run(cache_path, model, out_npy, batch=8, device="cuda"):
    arr, meta = pam_io.load_cache(cache_path)
    n = len(arr)
    out = np.zeros((n, arr.shape[1], arr.shape[2]), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            block = []
            for i in range(s, e):
                idx = [int(np.clip(i + k, 0, n - 1)) for k in (-2, -1, 0, 1, 2)]
                block.append(np.stack([np.asarray(arr[j], dtype=np.float32) for j in idx]))
            x = torch.from_numpy(np.stack(block)).to(device)
            out[s:e] = model(x).squeeze(1).cpu().numpy()
    np.save(out_npy, out)
    print(f"[infer] {out_npy} {out.shape}")
    return arr, out, meta


def figure(raw, enh, tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    zmax = raw.shape[1] * C.LS_PITCH_UM / 1000.0
    xmax = raw.shape[2] * C.LS_X_BIN_UM / 1000.0
    ext = [0, xmax, zmax, 0]
    vmax = float(np.percentile(raw, 99.9))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, img, t in zip(axes, [raw.max(0), enh.max(0)], ["raw", "enhanced"]):
        ax.imshow(img, cmap="magma", aspect="equal", extent=ext, vmin=0, vmax=vmax)
        ax.set_title(f"{tag} {t} XY-MIP"); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"result_{tag}_MIP.png"), dpi=200)
    plt.close(fig)

    # 选深度方向信息量最大的一帧做 XZ 对比
    score = np.array([float(np.asarray(enh[i]).std()) for i in range(0, len(enh), max(1, len(enh) // 40))])
    y = int(np.argmax(score)) * max(1, len(enh) // 40)
    y = min(y, len(enh) - 1)
    ext2 = [0, xmax, zmax, 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    axes[0].imshow(np.asarray(raw[y]), cmap="magma", aspect="equal", extent=ext2,
                   vmin=0, vmax=vmax)
    axes[0].set_title(f"{tag} raw XZ (frame {y})")
    axes[1].imshow(enh[y], cmap="magma", aspect="equal", extent=ext2,
                   vmin=0, vmax=float(np.percentile(enh, 99.9)))
    axes[1].set_title(f"{tag} enhanced XZ (frame {y})")
    for ax in axes:
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Z (mm)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"result_{tag}_XZ.png"), dpi=200)
    plt.close(fig)
    print(f"[fig] {tag} -> {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_v2.pth")
    ap.add_argument("--mouse", default="all", choices=["1", "4", "all"])
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AxialNAFNet().to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    mice = ["4", "1"] if args.mouse == "all" else [args.mouse]
    for m in mice:
        cache = ensure_cache(m, force=args.force_cache)
        out_npy = os.path.join(C.WORK_DIR, f"mouse{m}_enhanced_v2.npy")
        raw, enh, meta = run(cache, model, out_npy, batch=args.batch, device=device)
        figure(raw, enh, f"Mouse{m}", C.QC_DIR)


if __name__ == "__main__":
    main()
