# AR-PAM Z 轴复原（v2.1）

以光片为指引，用"物理退化引导的伪配对"训练 2.5D 反卷积网络，把声光的 Z 向有效 PSF
从 17 μm 压到约 9–10 μm（收掉离焦拖尾、分开 10–15 μm 微层）。不是把声光变成 2.1 μm。

## 目录

| 文件 | 作用 |
|---|---|
| `config.py` | 全部物理参数（μm 制）、扫描/窗口/奇偶、训练超参 |
| `pam_io.py` | txt 读取、**奇数帧 A-line 翻转**、整卷 gain、缓存构建 |
| `data_reader.py` | 光片读取（原始 npy 或 float16 训练网格） |
| `ls_grid.py` | 19 GB npy → 4.0 GB float16 训练网格 |
| `simulate.py` | 退化模型：高斯包络（默认）/ RF 域带通脉冲（`SIM_MODE="rf"`） |
| `dataset.py` / `model.py` / `train.py` | 2.5D 伪配对训练 |
| `infer.py` | 推理 + 出图 |
| `evaluate_real.py` | FWHM / ghost / 前向一致性（不需要配准） |
| `surface.py` / `evaluate_layers.py` | 脑表面检测、展平、深度分层评估 |
| `npy.py` | `.ims` → `.npy`（裁剪方向已按 h5py 实测修正） |
| `qc_report.py` / `check_three_things.py` | 数据体检 |
| `REVIEW_v2.md` | 完整方案、三个物理事实的判定证据、误判复盘 |

## 数据（不进库）

```
data/     4.ims 1.ims 4_subset.npy 1_subset.npy  4/*.txt 1/*.txt
work/     4_ls_grid.npy (4.0 GB)  4_pam_v2.npy (1.7 GB)  + .json
```

## 训练

```bash
python train.py --mouse 4 --epochs 60
python infer.py --ckpt best_v2.pth --mouse 4
python compare_ls_pam.py --mouse 4 --frames 150 300 450
python evaluate_real.py --raw work/4_pam_v2.npy --enh work/mouse4_enhanced_v2.npy
```

## 版本回退

```bash
git log --oneline --decorate        # 看历史
git diff v2.1 -- config.py          # 和基线比
git checkout v2.1 -- .              # 整棵回到 v2.1
git checkout <commit> -- model.py   # 只回一个文件
```

没有 git 时用快照（单文件、自带全部历史）：

```bash
git clone backup/ar_pam_v2.1.bundle repo_restore
```

## 关键数字

- 双向扫描：翻转奇数帧后相邻帧相关 0.93、血管重合 0.96（不翻 0.26 / 0.17）
- Y：1000 帧 = 1000 个独立位置，10 μm/帧，共 10 mm
- 轴向 PSF：σ 3.3 px（7.0 μm），FWHM 17.0 μm
- 零训练上限：包络 RL 5.0 px / 10.5 μm（1.63×）→ **网络及格线 ≤5.0 px**
