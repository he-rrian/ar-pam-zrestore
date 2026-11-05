"""
伪配对数据集（v2）。

2.5D 上下文按"物理距离"取，不再按"帧号"：
    PAM_CONTEXT_UM = (-40, -20, 0, 20, 40) um
    -> 光片层偏移 = offset_um / 2.1 = (-19, -10, 0, 10, 19)
v1 用的是 (-10,-5,0,5,10) 层 = ±21 um，那是按"PAM 帧距 10 um"设计的；
拆分奇偶后 PAM 帧距是 20 um，必须同步改，否则训练/推理的 Y 上下文对不上。

返回 (X, Y, C, meta)：
    X : (5, 457, 1000)  2.5D 伪声光输入
    Y : (1, 457, 1000)  光片监督（横向 PSF 已匹配）
    C : (1, 457, 1000)  无噪声退化图，用于物理一致性损失
    meta : (3,) sigma_z, sigma_x, att_exp
"""

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
from data_reader import LightsheetReader, load_or_compute_stats
from simulate import generate_pseudo_pam, target_scale_weight


class PseudoPairDataset(Dataset):
    def __init__(self, ims_path, y_center_indices, randomize=True, stats=None):
        self.ims_path = ims_path
        self.y_center_indices = np.asarray(y_center_indices, dtype=np.int64)
        self.randomize = randomize
        self.reader = None
        self.stats = stats if stats is not None else load_or_compute_stats(ims_path)

    def __len__(self):
        return len(self.y_center_indices)

    def __getitem__(self, idx):
        if self.reader is None:
            self.reader = LightsheetReader(self.ims_path, stats=self.stats)

        rng = np.random.default_rng()
        if self.randomize:
            sigma_z = float(rng.uniform(C.PSF_SIGMA_Z_MIN, C.PSF_SIGMA_Z_MAX))
            sigma_x = float(rng.uniform(C.PSF_SIGMA_X_MIN, C.PSF_SIGMA_X_MAX))
            att_exp = float(rng.uniform(C.ATT_EXP_MIN, C.ATT_EXP_MAX))
            speckle = float(rng.uniform(C.SPECKLE_SCALE_MIN, C.SPECKLE_SCALE_MAX))
            noise = float(rng.uniform(C.NOISE_SIGMA_MIN, C.NOISE_SIGMA_MAX))
            gain = float(rng.uniform(C.PSEUDO_GAIN_MIN, C.PSEUDO_GAIN_MAX))
        else:
            sigma_z, sigma_x, att_exp = C.PSF_SIGMA_Z, C.PSF_SIGMA_X, C.ATT_EXP
            speckle, noise = C.SPECKLE_SCALE, C.NOISE_SIGMA
            gain = 1.0

        y_center = int(self.y_center_indices[idx])
        channels, target, clean_center = [], None, None

        for off_layers in C.CONTEXT_LS_LAYERS:
            ls_xz = self.reader.get_xz(y_center + off_layers, 0)
            surface = None
            if C.ATT_RELATIVE_TO_SURFACE:
                surface = self.reader.surface_depth(y_center + off_layers, 0)
            x_lr, y_hr, clean = generate_pseudo_pam(
                ls_xz, rng=rng, sigma_z=sigma_z, sigma_x=sigma_x, att_exp=att_exp,
                speckle_scale=speckle, noise_sigma=noise, surface=surface, gain=gain,
            )
            channels.append(x_lr)
            if off_layers == 0:
                target, clean_center = y_hr, clean

        X = torch.from_numpy(np.stack(channels, axis=0))
        Y = torch.from_numpy(target[np.newaxis, ...])
        Cc = torch.from_numpy(clean_center[np.newaxis, ...])
        meta = torch.tensor([sigma_z, sigma_x, att_exp], dtype=torch.float32)
        if C.TARGET_SCALE_WEIGHT:
            W = torch.from_numpy(target_scale_weight(target)[np.newaxis, ...])
        else:
            W = torch.ones_like(Y)
        return X, Y, Cc, meta, W


def build_split():
    """训练/验证 Y 索引（按 Y 切分，避免近邻泄漏）。"""
    train_y = list(range(C.LS_Y_START + 15, C.TRAIN_Y_END, C.TRAIN_Y_STRIDE))
    val_y = list(range(C.VAL_Y_START, C.VAL_Y_END, C.VAL_Y_STRIDE))
    return train_y, val_y
