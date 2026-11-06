# 服务器关闭后的恢复指南

服务器（AutoDL bjb2）已关闭。以下内容**全部可本地重建**，无需从服务器取任何大数据。

## 已备份到本仓库（models/）

| 文件 | 说明 | 指标 |
|---|---|---|
| `ck_ep10.pth` | **最终定稿模型**（686 层，尺度修正 s114） | FWHM 15.5→10.2 μm，前向 0.35 |
| `ck_ep40.pth` | 上一版全深度模型（旧尺度 2.1 μm） | 10.9 μm，前向 0.44 |
| `backup_ck_ep40.pth` | 同上，作为回退点保存 | — |
| `best_v3.pth` | 最锐的 457 层窗口版本 | 8.6 μm，前向 0.51 |
| `best_fwhm.pth` | 中间版本（4.36 px） | — |
| `best_v2.pth` | 训练过程中的 val-best | — |

加载方式：
```python
from model import AxialNAFNet
m = AxialNAFNet(in_ch=6, base_ch=-1)
m.load_state_dict(torch.load("models/ck_ep10.pth", map_location="cpu", weights_only=True))
```

## 需要在本地重建的数据（脚本都在仓库里）

| 产物 | 重建命令 | 耗时 | 依赖 |
|---|---|---|---|
| `work/4_ls_grid.npy` (457 层) | `python ls_grid.py --mouse 4` | ~8 min | `data/4_subset.npy` |
| `work/4_ls_grid_full.npy` (686 层) | `python build_ls_grid_full.py` | ~12 min | `data/4.ims` |
| `work/4_pam_v2.npy` (457 px) | `python -c "import pam_io,config as C; pam_io.build_cache(C.PAM_M4_TXT_DIR, C.CACHE_M4_PATH)"` | ~2 min | `data/4/*.txt` |
| `work/4_pam_full686.npy` (686 px, 20:500) | `python work/build_full_cache.py`（`C.TARGET_Z=686`） | ~2 min | 同上 |
| `work/4_pam_full686_s114.npy`（尺度修正） | `python rescale_pam114.py` | ~1 min | 上一步产物 |
| `work/mouse4_enhanced_v2.npy` | `python infer.py --ckpt models/ck_ep10.pth --mouse 4` | ~5 min | 686 缓存 + 686 网格 |
| `work/mouse1_enhanced_v2.npy` | `python infer.py --ckpt models/best_v3.pth --mouse 1` | ~5 min | mouse1 的 457 缓存 |
| 完整视野 MIP | `python mip_from_ims.py 4`（/1） | ~25 min | `data/4.ims` / `1.ims` |

**大文件一律不备份**（均可重建，总共约 30 GB）。

## 环境

- Python 3.12 + torch 2.12.1+cu130（服务器）；本地无 GPU 时用 CPU 推理（`infer.py` 会自动选设备）
- 依赖：numpy / scipy / matplotlib / h5py / pillow / torch

## 关键参数（复现时注意）

| 参数 | 值 | 备注 |
|---|---|---|
| 光片网格层数 | 686（全深度）/ 457（窗口版） | |
| 声光采样窗口 | 20:500 → 686 px | s114 版再重采样到 1264 μm |
| 像素物理尺寸 | 1.842 μm（s114）/ 2.1 μm（旧） | 尺度修正的依据是血管间距比 1.14 |
| 轴向 PSF | σ=3.3 px（≈7 μm），FWHM 17 μm | 304 个孤立峰标定 |
| 2.5D 上下文 | 5 帧（±20/±10 μm） | |
| 输入通道 | 6（5 帧 + RL 反卷积先验） | |

## 结果图

`result/1` … `result/6` 保存了各轮训练的图与说明，`result/6` 是最终版。
