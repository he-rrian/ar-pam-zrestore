"""
光片 / 原始声光 / 增强声光 的三列对比（v2）。

注意：
  * PAM 已拆奇偶，1 帧 = 20 μm；光片 1 行 = 2.1 μm；映射用 config.pam_frame_to_ls_row()。
  * 两套数据的 FOV 未必配准，并排图只能看纹理/锐度，不能逐像素比较。
  * aspect='equal' + mm 坐标：不再把 10 mm × 0.96 mm 拉成两个不同的比例。

用法:
    python compare_ls_pam.py --mouse 4 --frames 100 250 400
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="4", choices=["1", "4"])
    ap.add_argument("--frames", type=int, nargs="+", default=[100, 250, 400])
    args = ap.parse_args()

    ims = C.IMS_M4_NPY if args.mouse == "4" else C.IMS_M1_NPY
    cache = C.CACHE_M4_PATH if args.mouse == "4" else C.CACHE_M1_PATH
    enh_path = os.path.join(C.WORK_DIR, f"mouse{args.mouse}_enhanced_v2.npy")

    raw, _ = pam_io.load_cache(cache)
    enh = np.load(enh_path) if os.path.exists(enh_path) else None
    n = len(raw)
    reader = LightsheetReader(ims, stats=load_or_compute_stats(ims))

    zmm = C.TARGET_Z * C.LS_PITCH_UM / 1000
    xmm = C.TARGET_X * C.LS_X_BIN_UM / 1000
    ext = [0, xmm, zmm, 0]

    cols = 3 if enh is not None else 2
    fig, axes = plt.subplots(len(args.frames), cols, figsize=(5.2 * cols, 3.2 * len(args.frames)),
                             squeeze=False)
    for r, fi in enumerate(args.frames):
        fi = int(np.clip(fi, 0, n - 1))
        ly = C.pam_frame_to_ls_row(fi)
        ls = reader.get_xz(ly + C.LS_Y_START, 0)
        vmax_ls = float(np.percentile(ls, 99.9))
        vmax_p = float(np.percentile(np.asarray(raw[fi]), 99.9))

        axes[r, 0].imshow(ls, cmap="magma", aspect="equal", extent=ext, vmin=0, vmax=vmax_ls)
        axes[r, 0].set_title(f"Light-sheet Y={ly} ({ly * C.LS_PITCH_UM / 1000:.2f} mm)")
        axes[r, 1].imshow(np.asarray(raw[fi]), cmap="magma", aspect="equal", extent=ext,
                          vmin=0, vmax=vmax_p)
        axes[r, 1].set_title(f"Raw PAM frame {fi} ({fi * C.PAM_FRAME_PITCH_UM / 1000:.2f} mm)")
        if enh is not None:
            axes[r, 2].imshow(enh[fi], cmap="magma", aspect="equal", extent=ext, vmin=0,
                              vmax=float(np.percentile(enh, 99.9)))
            axes[r, 2].set_title(f"Enhanced PAM frame {fi}")
        for c in range(cols):
            axes[r, c].set_xlabel("X (mm)")
            axes[r, c].set_ylabel("Z (mm)")

    fig.suptitle(f"Mouse{args.mouse}: LS vs PAM (parity-split, Z window {C.PAM_Z_START}:{C.PAM_Z_END})",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(C.QC_DIR, f"compare_LS_PAM_Mouse{args.mouse}.png")
    fig.savefig(out, dpi=180)
    print(f"saved {out}")
    print("⚠️ 若两套 FOV 未配准，这里只能比较'锐度/纹理'，不能逐像素比较。")


if __name__ == "__main__":
    main()
