"""
数据体检报告（v2.1，不需要 torch / GPU）。

一次跑完，把"上一轮踩过的坑"全部量化成图 + 数字：

  1. 双向扫描：相邻帧 直接 vs 翻转后 的相关/重合/残余位移 -> qc/qc_<m>_mirror.png
  2. 深度窗口：完整 500 个 RF sample + 新窗口线               -> qc/qc_<m>_window.png
  3. 归一化：  直方图、饱和/削顶统计、轴向剖面                -> qc/qc_<m>_norm.png
  4. Y 剖面：  光片 vs 声光（判断 Y 跨度/起点是否离谱）        -> qc/qc_<m>_yprofile.png
  5. 缓存自检：相邻帧相关、MIP、动态范围                     -> qc/qc_<m>_cache.png
  6. 文本汇总：qc/qc_<m>.txt

用法:
    python qc_report.py --mouse 4 --n 40
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import config as C
import pam_io


def _norm01(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)


def qc_mirror(txt_dir, mouse, n=6, outdir=C.QC_DIR):
    files = pam_io.list_frames(txt_dir)
    rep = pam_io.mirror_report(txt_dir)

    start = len(files) // 2
    start -= start % 2
    raw_frames = [pam_io.envelope(files[start + k], flip_x=False) for k in range(n)]
    fix_frames = [pam_io.envelope(files[start + k]) for k in range(n)]
    Cm = np.corrcoef(np.stack([s.ravel() for s in raw_frames]))

    r0, r1 = 40, 340
    c0, c1 = 250, 650
    crop = [s[r0:r1, c0:c1] for s in raw_frames]
    crop_fix = [s[r0:r1, c0:c1] for s in fix_frames]
    vmax = float(np.percentile(crop[0], 99.9))

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    im = ax[0, 0].imshow(Cm, cmap="viridis", vmin=0, vmax=1)
    ax[0, 0].set_title("frame-to-frame corr (raw, no flip) — checkerboard = mirroring")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046)

    ax[0, 1].imshow(crop[0], aspect="auto", cmap="magma", origin="upper", vmax=vmax)
    ax[0, 1].set_title(f"frame {pam_io.frame_index(files[start])} (even, forward)")
    ax[0, 2].imshow(crop[1], aspect="auto", cmap="magma", origin="upper", vmax=vmax)
    ax[0, 2].set_title(f"frame {pam_io.frame_index(files[start + 1])} (odd, backward, raw)")

    ax[1, 0].imshow(crop_fix[1], aspect="auto", cmap="magma", origin="upper", vmax=vmax)
    ax[1, 0].set_title("odd frame after A-line flip -> matches even")
    ax[1, 1].imshow(np.abs(crop[0] - crop_fix[1]), aspect="auto", cmap="inferno",
                    origin="upper", vmax=vmax * 0.8)
    ax[1, 1].set_title("|even - flipped odd| (residual = noise)")
    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 0.85,
                  "corr direct      %.3f\n"
                  "corr flipped     %.3f\n"
                  "overlap direct   %.2f\n"
                  "overlap flipped  %.2f\n"
                  "residual shift   dz=%d dx=%d\n\n"
                  "=> bidirectional raster scan:\n"
                  "   flip ODD frames along A-line."
                  % (rep["corr_direct"], rep["corr_flipped"],
                     rep["overlap_direct"], rep["overlap_flipped"],
                     rep["residual_shift_dz"], rep["residual_shift_dx"]),
                  fontsize=13, va="top", family="monospace")
    fig.tight_layout()
    p = os.path.join(outdir, f"qc_{mouse}_mirror.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return rep, p


def qc_window(txt_dir, mouse, n=2, outdir=C.QC_DIR):
    files = pam_io.list_frames(txt_dir)
    idx = np.linspace(0, len(files) - 1, n).astype(int)
    start_auto, end_auto, shallow = pam_io.estimate_window(files)

    fig, axes = plt.subplots(n, 2, figsize=(14, 3.6 * n), squeeze=False)
    for r, i in enumerate(idx):
        e = pam_io.envelope(files[i])
        vmax = np.percentile(e, 99.9)
        axes[r, 0].imshow(e, aspect="auto", cmap="magma", origin="upper", vmax=vmax)
        axes[r, 0].axhline(C.PAM_Z_START, color="cyan", ls="--", lw=1)
        axes[r, 0].axhline(C.PAM_Z_END, color="cyan", ls="--", lw=1)
        axes[r, 0].set_title(f"frame {pam_io.frame_index(files[i])}: full 500 RF "
                             f"(cyan = window {C.PAM_Z_START}:{C.PAM_Z_END})")
        axes[r, 0].set_xlabel("A-line"); axes[r, 0].set_ylabel("RF sample")

        p = gaussian_filter1d(e.mean(axis=1), 2.0)
        axes[r, 1].plot(p, np.arange(len(p)), lw=1)
        axes[r, 1].invert_yaxis()
        axes[r, 1].axhline(C.PAM_Z_START, color="cyan", ls="--")
        axes[r, 1].axhline(C.PAM_Z_END, color="cyan", ls="--")
        axes[r, 1].set_title("mean envelope vs RF sample"); axes[r, 1].grid(alpha=.3)

    fig.tight_layout()
    p = os.path.join(outdir, f"qc_{mouse}_window.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return {"window_start": int(start_auto), "window_end": int(end_auto),
            "shallow_surface_sample": float(shallow)}, p


def qc_norm(txt_dir, mouse, n=12, outdir=C.QC_DIR):
    files = pam_io.list_frames(txt_dir)
    cal = pam_io.calibrate(files)
    gain = cal["gain"]
    idx = np.linspace(0, len(files) - 1, n).astype(int)

    frames, bgs = [], []
    for i in idx:
        g, bg = pam_io.frame_to_grid(files[i])
        frames.append(g / gain)
        bgs.append(bg)
    frames = np.stack(frames)
    clip = pam_io.clipping_report(frames, hi=1.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].hist(frames.ravel(), bins=200, range=(0, 1.5), log=True)
    axes[0].axvline(1.0, color="r", ls="--")
    axes[0].set_title(f"normalized PAM (gain={gain:.4f})\n"
                      f"max={clip['max']:.2f}  frac>=1 {clip['frac_ge_hi'] * 100:.2f}%")
    axes[1].imshow(frames[n // 2], cmap="magma", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("single frame (training scale)")
    col = int(np.argmax(frames[n // 2].max(axis=0)))
    axes[2].plot(frames[n // 2][:, col], lw=1)
    axes[2].axhline(1.0, color="r", ls="--")
    axes[2].set_title(f"axial profile X={col} (no hard ceiling)")
    axes[3].imshow(frames.max(0), cmap="magma", aspect="auto", vmin=0, vmax=1)
    axes[3].set_title("max over sampled frames")
    fig.tight_layout()
    p = os.path.join(outdir, f"qc_{mouse}_norm.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return {"gain": gain, "bg_median": cal["bg_median"], "clip": clip}, p


def qc_yprofile(txt_dir, ls_npy, mouse, outdir=C.QC_DIR):
    files = pam_io.list_frames(txt_dir)
    idx = np.linspace(0, len(files) - 1, 40).astype(int)
    pam = np.array([pam_io.envelope(files[i]).mean() for i in idx])
    pam = gaussian_filter1d(pam, 3.0)

    vol = np.load(ls_npy, mmap_mode="r")
    ys = np.arange(0, vol.shape[1], 20)
    ls = np.array([float(np.mean(np.maximum(
        np.asarray(vol[::3, y, ::20], dtype=np.float32) - 419.0, 0))) for y in ys])
    ls = gaussian_filter1d(ls, 5.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.linspace(0, 1, len(pam)), _norm01(pam),
            label=f"PAM frame mean ({len(files)} frames)")
    ax.plot(np.linspace(0, 1, len(ls)), _norm01(ls), label="LS signal vs Y")
    ax.set_xlabel("normalized Y position"); ax.set_ylabel("normalized intensity")
    ax.set_title(f"mouse {mouse}: 1D Y profiles (sanity check, not registration)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"qc_{mouse}_yprofile.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return {"pam_frames": len(files), "ls_rows": int(vol.shape[1])}, p


def qc_cache(cache_path, mouse, outdir=C.QC_DIR):
    arr, meta = pam_io.load_cache(cache_path)
    n = len(arr)
    mid = n // 2
    idx = list(range(max(0, mid - 3), min(n, max(0, mid - 3) + 6)))
    S = [np.asarray(arr[i], dtype=np.float32) for i in idx]
    adj = [float(np.corrcoef(S[k].ravel(), S[k + 1].ravel())[0, 1]) for k in range(len(S) - 1)]

    mip_xy = np.asarray(arr.max(axis=1), dtype=np.float32)
    mip_xz = np.asarray(arr[max(0, mid - 5):mid + 6].max(axis=0), dtype=np.float32)
    mip_yz = np.asarray(arr.mean(axis=2), dtype=np.float32)

    zmm = arr.shape[1] * C.LS_PITCH_UM / 1000.0
    xmm = arr.shape[2] * C.LS_X_BIN_UM / 1000.0
    ymm = n * C.PAM_FRAME_PITCH_UM / 1000.0

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    ax[0, 0].imshow(mip_xy, cmap="magma", aspect="equal",
                    extent=[0, xmm, ymm, 0], vmin=0, vmax=np.percentile(mip_xy, 99.9))
    ax[0, 0].set_title(f"XY MIP ({n} frames = {ymm:.1f} mm)")
    ax[0, 0].set_xlabel("X (mm)"); ax[0, 0].set_ylabel("Y (mm)")

    ax[0, 1].imshow(mip_xz, cmap="magma", aspect="equal",
                    extent=[0, xmm, zmm, 0], vmin=0, vmax=np.percentile(mip_xz, 99.9))
    ax[0, 1].set_title(f"XZ MIP (frames {mid - 5}..{mid + 5})")
    ax[0, 1].set_xlabel("X (mm)"); ax[0, 1].set_ylabel("Z (mm)")

    ax[1, 0].imshow(mip_yz, cmap="magma", aspect="auto",
                    extent=[0, ymm, zmm, 0], vmin=0, vmax=np.percentile(mip_yz, 99.9))
    ax[1, 0].set_title("YZ mean projection")
    ax[1, 0].set_xlabel("Y (mm)"); ax[1, 0].set_ylabel("Z (mm)")

    ax[1, 1].axis("off")
    ax[1, 1].text(0.0, 0.9,
                  "cache: %s\nshape %s\nflip_odd_x %s (n_flipped %s)\n"
                  "adjacent-frame corr %.3f\nmax %.2f  p99.9 %.3f\nfrac>1 %.2f%%\n"
                  "Z window RF %d:%d"
                  % (os.path.basename(cache_path), arr.shape, meta.get("flip_odd_x"),
                     meta.get("n_flipped"), float(np.mean(adj)),
                     float(np.asarray(arr).max()), float(np.percentile(np.asarray(arr), 99.9)),
                     100 * float((np.asarray(arr) > 1).mean()),
                     meta.get("z_start", -1), meta.get("z_end", -1)),
                  fontsize=12, va="top", family="monospace")
    fig.tight_layout()
    p = os.path.join(outdir, f"qc_{mouse}_cache.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return {"n_frames": n, "adjacent_corr": float(np.mean(adj)),
            "max": float(np.asarray(arr).max()),
            "frac_gt1": float((np.asarray(arr) > 1).mean())}, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="4", choices=["1", "4"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--skip-ls", action="store_true")
    args = ap.parse_args()

    os.makedirs(C.QC_DIR, exist_ok=True)
    txt_dir = C.PAM_M4_TXT_DIR if args.mouse == "4" else C.PAM_M1_TXT_DIR
    ls_npy = C.IMS_M4_NPY if args.mouse == "4" else C.IMS_M1_NPY

    out = {"mouse": args.mouse, "config": C.summary()}

    rep, p = qc_mirror(txt_dir, args.mouse)
    out["mirror"] = rep
    print(f"[qc] mirror -> {p}\n{json.dumps(rep, indent=2, ensure_ascii=False)}")

    win, p = qc_window(txt_dir, args.mouse)
    out["window"] = win
    print(f"[qc] window -> {p} {win}")

    nrm, p = qc_norm(txt_dir, args.mouse, n=args.n)
    out["normalization"] = nrm
    print(f"[qc] norm -> {p} max={nrm['clip']['max']:.2f} "
          f"frac>=1={nrm['clip']['frac_ge_hi'] * 100:.2f}%")

    if not args.skip_ls and os.path.exists(ls_npy):
        yp, p = qc_yprofile(txt_dir, ls_npy, args.mouse)
        out["yprofile"] = yp
        print(f"[qc] yprofile -> {p}")

    cache = C.CACHE_M4_PATH if args.mouse == "4" else C.CACHE_M1_PATH
    if os.path.exists(cache):
        cc, p = qc_cache(cache, args.mouse)
        out["cache"] = cc
        print(f"[qc] cache -> {p} {cc}")

    tp = os.path.join(C.QC_DIR, f"qc_{args.mouse}.txt")
    with open(tp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[qc] summary -> {tp}")


if __name__ == "__main__":
    main()
