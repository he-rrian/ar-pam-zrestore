"""
check_three_things.py —— 三个物理事实边界的一次性自检。

  1. 奇偶：双向扫描（奇数帧 X 镜像）还是双波长/重复采集？
  2. Y 步长：1000 帧是 1000 个 Y 位置（10 μm）还是 500 个位置（20 μm）？
  3. 光片 457 层裁剪：脑表面有没有贴到 Z=0 上沿？

用法:
    python check_three_things.py --mouse 4
输出:
    qc/three_checks_<mouse>.png   qc/three_checks_<mouse>.json
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

import config as C
import pam_io
from data_reader import LightsheetReader, load_or_compute_stats


# ---------------------------------------------------------------------------
# 1. 奇偶
# ---------------------------------------------------------------------------
def check_parity(txt_dir):
    """返回 (MSE 直接/翻转, 相关, 重合, 是否双向扫描)。"""
    files = pam_io.list_frames(txt_dir)
    evens = [i for i in range(len(files) - 1)
             if pam_io.frame_index(files[i]) % 2 == 0 and pam_io.frame_index(files[i + 1]) % 2 == 1]
    picks = np.linspace(len(evens) // 4, len(evens) * 3 // 4, 6).astype(int)

    mse_d, mse_f, cd, cf, od, of = [], [], [], [], [], []
    for p in picks:
        e = evens[p]
        a = pam_io.envelope(files[e], flip_x=False)
        b = pam_io.envelope(files[e + 1], flip_x=False)
        bf = b[:, ::-1]
        mse_d.append(float(np.mean((a - b) ** 2)))
        mse_f.append(float(np.mean((a - bf) ** 2)))
        cd.append(pam_io._corr(a, b))
        cf.append(pam_io._corr(a, bf))
        od.append(pam_io._mask_overlap(a, b))
        of.append(pam_io._mask_overlap(a, bf))

    return {
        "mse_direct": float(np.mean(mse_d)),
        "mse_flipped": float(np.mean(mse_f)),
        "mse_ratio": float(np.mean(mse_f) / max(np.mean(mse_d), 1e-12)),
        "corr_direct": float(np.mean(cd)),
        "corr_flipped": float(np.mean(cf)),
        "overlap_direct": float(np.mean(od)),
        "overlap_flipped": float(np.mean(of)),
        "verdict": "bidirectional" if np.mean(of) > np.mean(od) + 0.2 else "same_position",
    }


# ---------------------------------------------------------------------------
# 2. Y 步长
# ---------------------------------------------------------------------------
def check_y_step(txt_dir, band=(350, 470)):
    """
    用"相关 vs 帧距"判断相邻帧是不是同一个 Y：
      * 若是"每个 Y 采两帧"（双波长/重复），d=1 的相关应接近 1（同一位置）；
      * 若是"每帧一个 Y"（往返扫描），d=1 只是一段平滑衰减的起点，明显 < 1。
    返回各帧距的平均相关。
    """
    files = pam_io.list_frames(txt_dir)
    lo, hi = band
    hi = min(hi, len(files))
    E = [pam_io.envelope(files[k]) for k in range(lo, hi)]   # 已按奇数帧翻转
    n = len(E)

    ds = [1, 2, 3, 4, 6, 8, 12, 16, 24]
    out = {}
    for d in ds:
        if d >= n:
            continue
        vals = [pam_io._corr(E[k], E[k + d]) for k in range(0, n - d, 2)]
        out[d] = float(np.mean(vals))
    return out, (lo, hi)


# ---------------------------------------------------------------------------
# 3. 光片顶边
# ---------------------------------------------------------------------------
def check_ls_top(ls_npy, y_step=50):
    """
    逐 Y 估计"脑表面前沿"（Z=0 为裁剪窗口顶部）：
      surf == 0 且 top_energy 不低  ->  表面贴顶/被截
    """
    reader = LightsheetReader(ls_npy, stats=load_or_compute_stats(ls_npy))
    ys = np.arange(0, C.LS_Y_ROWS, y_step)
    surf, top_e, peak_z = [], [], []
    for y in ys:
        img = reader.get_xz(y + C.LS_Y_START, 0)
        prof = np.percentile(img, 90, axis=1).astype(np.float32)
        prof = gaussian_filter1d(prof, 4.0)
        mx = float(prof.max() + 1e-9)
        s = int(np.argmax(prof > 0.05 * mx))
        surf.append(s)
        top_e.append(float(img[0:5].mean()))
        peak_z.append(int(np.argmax(prof)))
    surf = np.array(surf)
    return {
        "ys": ys.tolist(),
        "surface_z": surf.tolist(),
        "peak_z": peak_z,
        "top_energy": top_e,
        "n_y": int(len(ys)),
        "surf_min": int(surf.min()),
        "surf_median": float(np.median(surf)),
        "frac_touching": float(np.mean(surf <= 2)),
        "n_within_20": int(np.sum(surf < 20)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="4", choices=["1", "4"])
    args = ap.parse_args()

    txt_dir = C.PAM_M4_TXT_DIR if args.mouse == "4" else C.PAM_M1_TXT_DIR
    ls_npy = C.IMS_M4_NPY if args.mouse == "4" else C.IMS_M1_NPY

    print("=" * 72)
    print(f"1) 奇偶（mouse {args.mouse}）：MSE 与结构重合")
    par = check_parity(txt_dir)
    print(f"   MSE 直接      {par['mse_direct']:.6f}")
    print(f"   MSE 翻转奇数帧 {par['mse_flipped']:.6f}   (比值 {par['mse_ratio']:.3f})")
    print(f"   相关 直接/翻转 {par['corr_direct']:.3f} / {par['corr_flipped']:.3f}")
    print(f"   重合 直接/翻转 {par['overlap_direct']:.3f} / {par['overlap_flipped']:.3f}")
    print(f"   => {par['verdict']}")

    print("-" * 72)
    print("2) Y 步长：相关 vs 帧距（高 SNR 段）")
    curve, band = check_y_step(txt_dir)
    print(f"   帧段 {band}")
    for d, v in curve.items():
        print(f"   d={d:2d} 帧 ({d * C.PAM_FRAME_PITCH_UM:3.0f} um): corr {v:.3f}")
    d1 = curve.get(1, float("nan"))
    d2 = curve.get(2, float("nan"))
    print(f"   d=1 是否接近 1（同一 Y 的重复采集）？ {d1:.3f}")
    print("   => " + ("每帧是独立 Y（往返扫描），1000 帧 = 1000 个位置"
                     if d1 < 0.985 else "相邻帧疑似同一 Y 的重复采集"))

    print("-" * 72)
    print("3) 光片 457 层裁剪：脑表面深度分布")
    ls = check_ls_top(ls_npy)
    print(f"   采样 Y 数 {ls['n_y']}  表面 Z：min {ls['surf_min']}  median {ls['surf_median']:.0f}")
    print(f"   贴顶（surface<=2 层）占比 {ls['frac_touching'] * 100:.1f}%")
    print(f"   表面 < 20 层（42 um）的 Y 数 {ls['n_within_20']}/{ls['n_y']}")
    print("   => " + ("脑表面贴顶，需要从 .ims 向上重裁"
                     if ls["frac_touching"] > 0.05 or ls["n_within_20"] > ls["n_y"] * 0.2
                     else "上沿有足够背景，无需重裁"))

    # ---------------- 图 ----------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [par["corr_direct"], par["corr_flipped"]], "o-", label="corr")
    ax.plot([0, 1], [par["overlap_direct"], par["overlap_flipped"]], "s-", label="mask overlap")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["direct", "flip odd A-line"])
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"1) odd/even: {par['verdict']}\n"
                 f"MSE ratio (flip/direct) = {par['mse_ratio']:.3f}")

    ax = fig.add_subplot(gs[0, 1])
    ds = list(curve.keys()); vs = [curve[d] for d in ds]
    ax.plot(ds, vs, "o-")
    ax.axhline(1.0, color="r", ls="--", lw=1, label="corr=1 (same position)")
    ax.axhline(vs[0], color="gray", ls=":", lw=1, label=f"corr(d=1)={vs[0]:.3f}")
    ax.set_xlabel("frame distance d"); ax.set_ylabel("mean corr")
    ax.set_ylim(0, 1.05); ax.grid(alpha=.3); ax.legend()
    ax.set_title("2) corr vs frame distance (after mirror correction)")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(ls["ys"], np.array(ls["surface_z"]), lw=.8)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.axhline(20, color="orange", ls=":", lw=1)
    ax.set_xlabel("LS npy Y row"); ax.set_ylabel("surface Z (layers)")
    ax.set_title("3) LS brain surface depth vs Y   (0 = crop top; red dashed = clipped)")
    ax.grid(alpha=.3)

    plt.tight_layout()
    p = os.path.join(C.QC_DIR, f"three_checks_{args.mouse}.png")
    plt.savefig(p, dpi=150)
    out = {"mouse": args.mouse, "parity": par, "y_step": curve, "y_band": list(band),
           "ls_top": {k: v for k, v in ls.items() if k not in ("ys", "surface_z", "top_energy")}}
    jp = os.path.join(C.QC_DIR, f"three_checks_{args.mouse}.json")
    json.dump(out, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("-" * 72)
    print(f"saved {p}\n      {jp}")


if __name__ == "__main__":
    main()
