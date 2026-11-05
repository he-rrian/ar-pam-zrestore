"""
训练前自检（v2，不需要 torch）：光片 target / 伪声光 / 真实声光是否在同一分布。

和 v1 的 `verify_preprocess.py` 的区别：
  * 真实声光从 v2 缓存读（拆好奇偶、无上限 clip），不再现场 loadtxt；
  * 检查项加上"是否削顶"和"2.5D 上下文跨度"。

用法:
    python verify_preprocess.py --mouse 4
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pam_io
from data_reader import LightsheetReader, load_or_compute_stats
from simulate import generate_pseudo_pam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="4", choices=["1", "4"])
    ap.add_argument("--y", type=int, default=2350, help="光片 npy 的 Y 行号")
    args = ap.parse_args()

    ims = C.IMS_M4_NPY if args.mouse == "4" else C.IMS_M1_NPY
    cache = C.CACHE_M4_PATH if args.mouse == "4" else C.CACHE_M1_PATH

    st = load_or_compute_stats(ims)
    reader = LightsheetReader(ims, stats=st)
    print(f"[LS] bg={st['bg']:.1f} gain={st['gain']:.1f} col_std={st['col_profile'].std():.2f}")

    ls = reader.get_xz(args.y + C.LS_Y_START, 0)
    pseudo, target, clean = generate_pseudo_pam(
        ls, rng=np.random.default_rng(0),
        sigma_z=C.PSF_SIGMA_Z, sigma_x=C.PSF_SIGMA_X, att_exp=C.ATT_EXP,
    )

    pam = None
    if os.path.exists(cache):
        arr, meta = pam_io.load_cache(cache)
        pam = np.asarray(arr[len(arr) // 2], dtype=np.float32)
        print(f"[PAM] {cache} shape={arr.shape} flip_odd_x={meta.get('flip_odd_x')} "
              f"window={meta.get('z_start')}:{meta.get('z_end')} gain={meta.get('gain'):.5f}")
    else:
        print(f"[PAM] 没有缓存 {cache}，先跑 pam_io.build_cache(...)")

    print(f"{'':10s} {'mean':>8s} {'p50':>8s} {'p99':>8s} {'max':>8s} {'gradZ':>8s} {'gradX':>8s} {'>1':>8s}")
    for name, img in [("LS target", target), ("pseudo", pseudo), ("real PAM", pam)]:
        if img is None:
            continue
        print(f"{name:10s} {img.mean():8.4f} {np.median(img):8.4f} {np.percentile(img, 99):8.4f} "
              f"{img.max():8.4f} {np.abs(np.diff(img, axis=0)).mean():8.4f} "
              f"{np.abs(np.diff(img, axis=1)).mean():8.4f} {100 * (img > 1).mean():7.3f}%")

    ncol = 4 if pam is not None else 3
    fig, ax = plt.subplots(2, ncol, figsize=(5 * ncol, 8))
    imgs = [target, pseudo] + ([pam] if pam is not None else [])
    # 只显示中央 2.5 mm：10 mm x 0.96 mm 的整条在 equal aspect 下只有几毫米高
    x0, x1 = 375, 625
    for a, img, t in zip(ax[0], imgs, ["LS target", "pseudo-PAM", "real PAM"][:len(imgs)]):
        a.imshow(img[:, x0:x1], cmap="magma", aspect="equal", vmin=0, vmax=1,
                 extent=[x0 * C.LS_X_BIN_UM / 1000, x1 * C.LS_X_BIN_UM / 1000,
                         C.TARGET_Z * C.LS_PITCH_UM / 1000, 0])
        a.set_title(t); a.set_xlabel("X (mm)"); a.set_ylabel("Z (mm)")

    ax[1, 0].hist(target.ravel(), bins=120, range=(0, 1.5), alpha=.6, label="LS target", log=True)
    ax[1, 0].hist(pseudo.ravel(), bins=120, range=(0, 1.5), alpha=.6, label="pseudo", log=True)
    if pam is not None:
        ax[1, 0].hist(pam.ravel(), bins=120, range=(0, 1.5), alpha=.6, label="real PAM", log=True)
    ax[1, 0].axvline(1.0, color="r", ls="--")
    ax[1, 0].legend(); ax[1, 0].set_title("histograms (red = old hard clip)")

    col = int(np.argmax(target.max(axis=0)))
    ax[1, 1].plot(target[:, col], label="LS target", lw=1)
    ax[1, 1].plot(pseudo[:, col], label="pseudo", lw=1)
    if pam is not None:
        ax[1, 1].plot(pam[:, col], label="real PAM", lw=1)
    ax[1, 1].legend(); ax[1, 1].set_title(f"axial profile X={col}")

    ax[1, 2].imshow(np.maximum(target - pseudo, 0), cmap="magma", aspect="auto")
    ax[1, 2].set_title("target - pseudo (positive)")

    if ncol == 4:
        nb = reader.get_xz(args.y + C.CONTEXT_LS_LAYERS[0] + C.LS_Y_START, 0)
        ax[0, 3].imshow(nb[:, x0:x1], cmap="magma", aspect="equal", vmin=0, vmax=1,
                        extent=[x0 * C.LS_X_BIN_UM / 1000, x1 * C.LS_X_BIN_UM / 1000,
                                C.TARGET_Z * C.LS_PITCH_UM / 1000, 0])
        ax[0, 3].set_title(f"2.5D neighbour ({C.PAM_CONTEXT_UM[0]:.0f} um)")
        ax[0, 3].set_xlabel("X (mm)"); ax[0, 3].set_ylabel("Z (mm)")
        ax[1, 3].imshow(np.abs(target[:, x0:x1] - pam[:, x0:x1]) if pam is not None else np.zeros((2, 2)),
                        cmap="magma", aspect="equal",
                        extent=[x0 * C.LS_X_BIN_UM / 1000, x1 * C.LS_X_BIN_UM / 1000,
                                C.TARGET_Z * C.LS_PITCH_UM / 1000, 0])
        ax[1, 3].set_title("|LS target - real PAM| (not registered, only texture)")
    plt.tight_layout()
    p = os.path.join(C.QC_DIR, f"verify_preprocess_{args.mouse}.png")
    plt.savefig(p, dpi=150)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
