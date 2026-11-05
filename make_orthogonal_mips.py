"""
正交 MIP（v2.1）：LS 与 PAM 的 XZ / YZ 全深度视图，等比例物理坐标。

v1 的问题：`aspect='auto'` 把 10 mm x 0.96 mm 塞进不匹配的面板，看起来
一个"宽扁"一个"窄长"；且 PAM 用的是未纠偏（奇偶镜像混入）的 1000 帧。

用法:
    python make_orthogonal_mips.py --mouse 4
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io
from data_reader import load_or_compute_stats


def ls_mips(path):
    a = np.load(path, mmap_mode="r")
    Zs, Ys, Xs = a.shape
    if Xs == C.TARGET_X:                     # 已经是训练网格（float16，已归一化）
        return a.max(axis=1).astype(np.float32), a.mean(axis=2).astype(np.float32)

    st = load_or_compute_stats(path)
    edges = np.linspace(0, Xs, C.TARGET_X + 1).astype(np.int64)
    counts = np.diff(edges).astype(np.float32)

    mip_xz = np.zeros((Zs, C.TARGET_X), np.float32)
    mip_yz = np.zeros((Zs, Ys), np.float32)
    for y0 in range(0, Ys, 25):
        y1 = min(y0 + 25, Ys)
        block = np.asarray(a[:, y0:y1, :], dtype=np.float32)
        block -= st["bg"]
        np.maximum(block, 0, out=block)
        sums = np.add.reduceat(block, edges[:-1], axis=2)
        mip_xz = np.maximum(mip_xz, (sums / counts[None, None, :]).max(axis=1))
        mip_yz[:, y0:y1] = block.max(axis=2)
    g = st["gain"]
    return mip_xz / g, mip_yz / g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="4", choices=["1", "4"])
    args = ap.parse_args()

    ims = C.IMS_M4_NPY if args.mouse == "4" else C.IMS_M1_NPY
    cache = C.CACHE_M4_PATH if args.mouse == "4" else C.CACHE_M1_PATH
    if not os.path.exists(cache):
        raise SystemExit(f"没有缓存 {cache}，先跑 pam_io.build_cache(...)")

    mip_xz, mip_yz = ls_mips(ims)
    pam, meta = pam_io.load_cache(cache)
    n = len(pam)
    pam_xz = np.asarray(pam.max(axis=0), dtype=np.float32)
    pam_yz = np.asarray(pam.mean(axis=2), dtype=np.float32)

    zmm = C.TARGET_Z * C.LS_PITCH_UM / 1000
    xmm = C.TARGET_X * C.LS_X_BIN_UM / 1000
    ymm_ls = mip_yz.shape[1] * C.LS_PITCH_UM / 1000
    ymm_pam = n * C.PAM_FRAME_PITCH_UM / 1000

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    ax[0, 0].imshow(mip_xz, cmap="magma", aspect="equal", extent=[0, xmm, zmm, 0],
                    vmin=0, vmax=np.percentile(mip_xz, 99.5))
    ax[0, 0].set_title(f"LS XZ MIP (all Y = {ymm_ls:.1f} mm)")
    ax[0, 0].set_xlabel("X (mm)"); ax[0, 0].set_ylabel("Z (mm)")

    ax[0, 1].imshow(pam_xz, cmap="magma", aspect="equal", extent=[0, xmm, zmm, 0],
                    vmin=0, vmax=np.percentile(pam_xz, 99.5))
    ax[0, 1].set_title(f"PAM XZ MIP (all {n} frames = {ymm_pam:.1f} mm, mirror-corrected)")
    ax[0, 1].set_xlabel("X (mm)"); ax[0, 1].set_ylabel("Z (mm)")

    ax[1, 0].imshow(mip_yz, cmap="magma", aspect="auto", extent=[0, ymm_ls, zmm, 0],
                    vmin=0, vmax=np.percentile(mip_yz, 99.5))
    ax[1, 0].set_title("LS YZ MIP (all X)")
    ax[1, 0].set_xlabel("Y (mm)"); ax[1, 0].set_ylabel("Z (mm)")

    ax[1, 1].imshow(pam_yz.T, cmap="magma", aspect="auto", extent=[0, ymm_pam, zmm, 0],
                    vmin=0, vmax=np.percentile(pam_yz, 99.5))
    ax[1, 1].set_title("PAM YZ MIP (mean over X, mirror-corrected)")
    ax[1, 1].set_xlabel("Y (mm)"); ax[1, 1].set_ylabel("Z (mm)")

    plt.tight_layout()
    out = os.path.join(C.QC_DIR, f"fig_orthogonal_mips_{args.mouse}.png")
    plt.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
