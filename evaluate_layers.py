"""
§8 深度分层评估（方式 A/B 都支持）：以"软脑膜表面下多少 μm"为深度轴。

不需要 LS–PAM 逐像素配准：两条曲线都是分布级的（血管密度 vs 表面下深度）。

  * 光片：逐 Y 检 pia -> 只保留"表面完整落在窗口内"的 Y（方式 A）-> 展平
  * 声光：逐帧检表面回波 -> 展平
  * 比较：LS / PAM raw / PAM enhanced 的 density–depth 曲线相关

用法:
    python evaluate_layers.py --mouse 4
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io
import surface as S
from data_reader import LightsheetReader, load_or_compute_stats


def load_ls_flat(get_img, n=60, out_z=320):
    """get_img(y) -> (Z, X) 光片切片（背景已扣）。返回展平体、曲线、pia、有效 Y。"""
    ys = np.linspace(100, C.LS_Y_ROWS - 100, n).astype(int)
    flats, curves, ok_y, pias, examples = [], [], [], [], []
    for y in ys:
        img = get_img(int(y))
        pia = S.pia_from_ls(img)
        m = S.valid_columns(pia, img.shape[0], need=min(out_z, 120), colmax=img.max(axis=0))
        pias.append((y, pia, float(m.mean())))
        if m.mean() < 0.6:          # 表面大部分不在窗口内，跳过
            continue
        if len(examples) < 3:
            examples.append((int(y), img, pia))
        f = S.flatten(img, pia, out_z=out_z)
        f[:, ~m] = 0
        flats.append(f)
        curves.append(S.depth_curve(f))
        ok_y.append(y)
    return (np.stack(flats) if flats else None), curves, pias, ys, np.array(ok_y), examples


def load_pam_flat(cache_path, n=80, out_z=320):
    arr, meta = pam_io.load_cache(cache_path)
    idx = np.linspace(0, len(arr) - 1, min(n, len(arr))).astype(int)
    flats, curves = [], []
    for i in idx:
        img = np.asarray(arr[i], dtype=np.float32)
        surf = S.surface_from_pam(img)
        f = S.flatten(img, surf, out_z=out_z)
        f[:, ~np.isfinite(surf)] = 0
        flats.append(f)
        curves.append(S.depth_curve(f))
    return np.stack(flats), curves, idx


def fwhm_vs_depth(pam_cache, enh=None, depth_edges=(0, 100, 200, 320), out_z=320):
    """把每帧展平后，按表面下深度分箱，统计孤立峰 Z-FWHM 的中位数。"""
    from scipy.signal import find_peaks

    def _fwhm(col, k, half=8):
        if k < half or k > len(col) - half - 1:
            return np.nan
        y = col[k - half:k + half + 1]
        if y.max() <= 1e-6:
            return np.nan
        h = y.max() / 2.0
        i = int(np.argmax(y))
        l = i
        while l > 0 and y[l] > h:
            l -= 1
        r = i
        while r < len(y) - 1 and y[r] > h:
            r += 1
        if l == 0 or r == len(y) - 1:
            return np.nan
        fl = (y[l + 1] - h) / (y[l + 1] - y[l] + 1e-9)
        fr = (y[r - 1] - h) / (y[r - 1] - y[r] + 1e-9)
        return float((r - fr) - (l + fl))

    arr, _ = pam_io.load_cache(pam_cache)
    idx = np.linspace(0, len(arr) - 1, min(120, len(arr))).astype(int)
    bins = {f"{depth_edges[k]}-{depth_edges[k+1]}": [[], []] for k in range(len(depth_edges) - 1)}
    for i in idx:
        raw = np.asarray(arr[i], dtype=np.float32)
        s = S.surface_from_pam(raw)
        fr = S.flatten(raw, s, out_z=out_z)
        fe = S.flatten(np.asarray(enh[i], dtype=np.float32), s, out_z=out_z) if enh is not None else None
        for k in range(len(depth_edges) - 1):
            a, b = depth_edges[k], depth_edges[k + 1]
            key = f"{a}-{b}"
            for c in range(0, fr.shape[1], 10):
                col = fr[a:b, c]
                if col.max() <= 0:
                    continue
                pk, _ = find_peaks(col, prominence=0.2 * col.max(), distance=6)
                for p in pk:
                    v = _fwhm(col, p, half=8)
                    if np.isfinite(v) and 2 < v < 30:
                        bins[key][0].append(v)
                        if fe is not None:
                            v2 = _fwhm(fe[a:b, c], p, half=8)
                            if np.isfinite(v2):
                                bins[key][1].append(v2)
    return {k: (float(np.median(v[0])) if v[0] else np.nan,
                float(np.median(v[1])) if v[1] else np.nan) for k, v in bins.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="4", choices=["1", "4"])
    ap.add_argument("--n_ls", type=int, default=60)
    ap.add_argument("--n_pam", type=int, default=80)
    ap.add_argument("--out_z", type=int, default=320, help="展平后的深度层数（1 层 = 2.1 um）")
    ap.add_argument("--ims", action="store_true",
                    help="直接从 .ims 读整段 Z（方式 B，需要 h5py；不受 457 层窗口限制）")
    ap.add_argument("--z0", type=int, default=1100, help="--ims 时的 Z 起点")
    args = ap.parse_args()

    ims = C.IMS_M4_NPY if args.mouse == "4" else C.IMS_M1_NPY
    ims_raw = os.path.join(C.DATA_DIR, f"{args.mouse}.ims")
    cache = C.CACHE_M4_PATH if args.mouse == "4" else C.CACHE_M1_PATH
    enh_path = os.path.join(C.WORK_DIR, f"mouse{args.mouse}_enhanced_v2.npy")

    if args.ims:
        rd = S.ImsReader(ims_raw)
        flip = C.LS_CROP[args.mouse]["flip_z"]

        def get_img(y):
            img = rd.slice_xz(y + C.LS_Y_START, args.z0, None, flip_z=flip)
            return np.maximum(img - float(np.median(img)), 0.0)

        print(f"[LS] 直接从 {os.path.basename(ims_raw)} 读 Z[{args.z0}:{rd.Z}] flip={flip}")
    else:
        reader = LightsheetReader(ims, stats=load_or_compute_stats(ims))
        print(f"[LS] 从 {os.path.basename(ims)} 读 457 层窗口")

        def get_img(y):
            return reader.get_xz(y + C.LS_Y_START, 0)

    ls_flat, ls_curves, pias, ys, ok_y, examples = load_ls_flat(get_img, args.n_ls, args.out_z)
    print(f"[LS] 采样 {len(ys)} 个 Y，表面完整落在窗口内的 {len(ok_y)} 个 "
          f"({100 * len(ok_y) / len(ys):.0f}%)")
    print("     有效 Y 范围:", (ok_y.min(), ok_y.max()) if len(ok_y) else "无")

    pam_flat, pam_curves, pam_idx = load_pam_flat(cache, args.n_pam, args.out_z)
    enh = np.load(enh_path, mmap_mode="r") if os.path.exists(enh_path) else None

    def curve(cs, key="density"):
        return np.mean([c[key] for c in cs], axis=0) if cs else np.zeros(args.out_z)

    cl = curve(ls_curves)
    cp = curve(pam_curves)
    n = args.out_z
    r = float(np.corrcoef(S.resample_curve(cl, 200), S.resample_curve(cp, 200))[0, 1])
    print(f"[curve] LS vs PAM(raw) density–depth 相关 = {r:.3f}")

    print("[FWHM vs 表面下深度]  median Z-FWHM (px, 1 px = 2.1 um)")
    fw = fwhm_vs_depth(cache, enh, (0, 100, 200, 320), args.out_z)
    for k, (a, b) in fw.items():
        print(f"   {k:>8} 层: raw {a:5.1f}" + (f"  enh {b:5.1f}" if enh is not None else ""))

    # ---------------- 图 ----------------
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    yy = [p[0] for p in pias]
    mm = [p[2] for p in pias]
    axs[0, 0].plot(yy, mm, lw=.9)
    axs[0, 0].set_title(f"mouse{args.mouse}: fraction of columns with pia inside the 457-slab")
    axs[0, 0].set_xlabel("LS npy Y"); axs[0, 0].set_ylabel("valid fraction")
    axs[0, 0].set_ylim(-0.02, 1.02); axs[0, 0].grid(alpha=.3)

    zz = np.arange(n) * C.LS_PITCH_UM
    axs[0, 1].plot(zz, cl / (cl.max() + 1e-9), label="LS")
    axs[0, 1].plot(zz, cp / (cp.max() + 1e-9), label="PAM raw")
    axs[0, 1].set_xlabel("depth below pia (um)"); axs[0, 1].set_ylabel("normalized vessel density")
    axs[0, 1].set_title(f"density–depth (corr {r:.2f})"); axs[0, 1].legend(); axs[0, 1].grid(alpha=.3)

    if examples:
        y0, img0, pia0 = examples[0]
        axs[1, 0].imshow(img0, aspect="equal", cmap="magma",
                         extent=[0, C.TARGET_X * C.LS_X_BIN_UM / 1000, img0.shape[0] * C.LS_PITCH_UM / 1000, 0],
                         vmin=0, vmax=np.percentile(img0, 99.9))
        axs[1, 0].plot(np.linspace(0, C.TARGET_X * C.LS_X_BIN_UM / 1000, img0.shape[1]),
                       pia0 * C.LS_PITCH_UM / 1000, color="cyan", lw=1)
        axs[1, 0].set_title(f"LS Y={y0}: detected pia (cyan) — check this first")
        axs[1, 0].set_xlabel("X (mm)"); axs[1, 0].set_ylabel("Z (mm)")
    axs[1, 1].imshow(pam_flat[len(pam_flat) // 2], aspect="equal", cmap="magma",
                     extent=[0, pam_flat.shape[2] * C.LS_X_BIN_UM / 1000, n * C.LS_PITCH_UM / 1000, 0],
                     vmin=0, vmax=np.percentile(pam_flat, 99.9))
    axs[1, 1].set_title("PAM flattened to depth-below-surface")
    axs[1, 1].set_xlabel("X (mm)"); axs[1, 1].set_ylabel("depth (mm)")

    plt.tight_layout()
    p = os.path.join(C.QC_DIR, f"layers_mouse{args.mouse}.png")
    plt.savefig(p, dpi=150)
    print(f"saved {p}")


if __name__ == "__main__":
    main()

