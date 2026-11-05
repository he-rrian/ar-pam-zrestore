"""
训练 v2：物理退化引导的伪配对 2.5D 反卷积。

用法（服务器上）：
    python -c "import pam_io, config as C; pam_io.build_cache(C.PAM_M4_TXT_DIR, C.CACHE_M4_PATH)"
    python train.py --mouse 4 --epochs 60

与 v1 的差别：
  * 真实数据探针直接从 v2 缓存读（v1 每个 epoch 都去 loadtxt 上千个 txt，
    train.log 里那一堆"成功提取真实声学背景噪声"就是它，极慢且和训练无关）；
  * 数据侧 2.5D 偏移改成物理距离（见 dataset.py）；
  * 仍然用"真实探针"监控坍缩/过曝，但不参与选模型——选模型只看伪验证集，
    真实数据的好坏必须靠 evaluate_real.py 的尺度无关指标。
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
import pam_io
from dataset import PseudoPairDataset, build_split
from model import AxialNAFNet, total_loss, physical_forward


def worker_init_fn(worker_id):
    np.random.seed(np.random.SeedSequence().entropy % (2 ** 32) + worker_id)


def build_real_probe(cache_path, n_ctx=5):
    """固定取缓存中间 5 帧做真实域探针（1,5,Z,X）；开 RL 先验时补第 6 通道。"""
    arr, _ = pam_io.load_cache(cache_path)
    mid = len(arr) // 2
    idx = [int(np.clip(mid + k, 0, len(arr) - 1)) for k in (-2, -1, 0, 1, 2)][:n_ctx]
    arrs = np.stack([np.asarray(arr[i], dtype=np.float32) for i in idx])
    if C.USE_RL_PRIOR:
        from simulate import rl_deconv
        rl = rl_deconv(arrs[len(arrs) // 2], C.PSF_SIGMA_Z)[None]
        arrs = np.concatenate([arrs, rl], axis=0)
    return torch.from_numpy(arrs[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="4", help="用哪只鼠的光片训练（4 或 1）")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--out", default="best_v2.pth")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ims = C.IMS_M4_NPY if args.mouse == "4" else C.IMS_M1_NPY
    cache = C.CACHE_M4_PATH if args.mouse == "4" else C.CACHE_M1_PATH

    train_y, val_y = build_split()
    train_ds = PseudoPairDataset(ims, train_y, randomize=True)
    val_ds = PseudoPairDataset(ims, val_y, randomize=False, stats=train_ds.stats)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  ims={os.path.basename(ims)}")
    print(C.summary())

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              worker_init_fn=worker_init_fn, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=max(1, args.workers // 2), pin_memory=True,
                            worker_init_fn=worker_init_fn,
                            persistent_workers=args.workers > 0)

    model = AxialNAFNet(in_ch=6 if C.USE_RL_PRIOR else 5,
                        base_ch=-1 if C.USE_RL_PRIOR else 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    probe = None
    if not args.no_probe and os.path.exists(cache):
        probe = build_real_probe(cache).to(device)

    best = float("inf")
    best_real = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for X, Y, Cg, meta, W in train_loader:
            X, Y, Cg, meta, W = [t.to(device, non_blocking=True) for t in (X, Y, Cg, meta, W)]
            pred = model(X)
            loss = total_loss(
                pred, Y,
                raw_input=X[:, 2:3],
                clean_deg=Cg,
                meta=meta,
                weight=W,
                lam_mean=C.LAM_MEAN, lam_grad=C.LAM_GRAD, lam_laplace=C.LAM_LAPLACE,
                lam_physics=C.LAM_PHYSICS, lam_bg=C.LAM_BG, lam_tv=C.LAM_TV, lam_thin=C.LAM_THIN,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            tl += float(loss.item())
        sched.step()
        tl /= max(1, len(train_loader))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, Y, Cg, meta, W in val_loader:
                X, Y, Cg, meta, W = [t.to(device, non_blocking=True) for t in (X, Y, Cg, meta, W)]
                pred = model(X)
                vl += float(total_loss(pred, Y, raw_input=X[:, 2:3], clean_deg=Cg,
                                       meta=meta, weight=W, lam_mean=C.LAM_MEAN, lam_grad=C.LAM_GRAD,
                                       lam_laplace=C.LAM_LAPLACE, lam_physics=C.LAM_PHYSICS,
                                       lam_bg=C.LAM_BG, lam_tv=C.LAM_TV).item())
        vl /= max(1, len(val_loader))

        msg = f"Epoch [{ep:02d}/{args.epochs}] train {tl:.5f} | val {vl:.5f}"

        # 真实域探针：既监控坍缩，也给一个"真实数据自洽性"分数做第二 checkpoint。
        # 注意：这个分数只衡量前向一致性与空白区，不能代表"重建正确"，
        # 所以另存为 best_real_*.pth，不覆盖按伪验证集选出的主模型。
        real_score = None
        if probe is not None:
            with torch.no_grad():
                out = model(probe)
                inp = probe[:, 2:3]
                empty = inp < 0.02
                meta_cal = torch.tensor([[C.PSF_SIGMA_Z, C.PSF_SIGMA_X, C.ATT_EXP]],
                                        device=device, dtype=out.dtype)
                deg = physical_forward(out, meta_cal)
                alpha = (deg * inp).sum() / ((deg * deg).sum() + 1e-8)
                res = ((alpha * deg - inp).norm() / (inp.norm() + 1e-8)).item()
                ghost = (out[empty].mean() if empty.any() else torch.tensor(0.)).item()
                real_score = res + 5.0 * ghost
                msg += (f" | real in {inp.mean():.3f} -> out {out.mean():.3f}"
                        f" fwd {res:.3f} ghost {ghost:.4f} score {real_score:.3f}"
                        f" sat>{(out > 1.2).float().mean() * 100:.2f}%")
        print(msg, flush=True)

        if vl < best:
            best = vl
            torch.save(model.state_dict(), args.out)
            print(f"  >>> saved {args.out} (val {best:.5f})", flush=True)

        if real_score is not None and ep > 10 and real_score < best_real:
            best_real = real_score
            torch.save(model.state_dict(), os.path.splitext(args.out)[0] + "_real.pth")
            print(f"  >>> saved {os.path.splitext(args.out)[0]}_real.pth "
                  f"(real score {best_real:.4f})", flush=True)

    torch.save(model.state_dict(), os.path.splitext(args.out)[0] + "_last.pth")


if __name__ == "__main__":
    main()
