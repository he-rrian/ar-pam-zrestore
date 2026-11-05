import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialBlock(nn.Module):
    """非对称 9x3 卷积 + 残差（保持原名）"""

    def __init__(self, c, kz=9, kx=3, dilation=1):
        super().__init__()
        pad_z = (kz // 2) * dilation
        pad_x = kx // 2
        self.conv1 = nn.Conv2d(
            c, c, (kz, kx), padding=(pad_z, pad_x), dilation=(dilation, 1)
        )
        self.norm = nn.GroupNorm(4, c)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.act(self.norm(self.conv1(x))))


class AxialNAFNet(nn.Module):
    """
    与旧版的区别：
      * 去掉 sigmoid(res_weight) 的 0.5 残差（它让网络一开始就"抄输入"，
        并把输出整体压到 0.5 倍，目标背景又被错误归一化到 0.16，
        两者叠加就是"黑底 + 高光"的来源之一）；
      * 改成 out = relu(x_center + head(h))，head 零初始化，
        初始时严格等于输入，之后只学"残差修正"；
      * 去掉上界 clamp(…,1.0)，避免血管峰被削平。
    """

    def __init__(self, in_ch=5, base=32):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        dilations = [1, 2, 4, 1, 2, 4]
        self.blocks = nn.Sequential(
            *[AxialBlock(base, kz=9, kx=3, dilation=d) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base // 2, 1, 3, padding=1),
        )
        # 零初始化最后一层 -> 初始输出严格等于输入
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return F.relu(x[:, 2:3, :, :] + self.head(h))


def make_psf_kernel(sigma_z, sigma_x, kz=41, kx=11):
    """
    sigma_z, sigma_x: (B,) 张量
    返回 (B, 1, kz, kx) 的可分离高斯 PSF，用于 groups=B 的逐样本卷积。
    """
    b = sigma_z.shape[0]
    device = sigma_z.device
    z = torch.arange(kz, device=device, dtype=torch.float32) - kz // 2
    x = torch.arange(kx, device=device, dtype=torch.float32) - kx // 2

    gz = torch.exp(-(z[None, :] ** 2) / (2.0 * sigma_z[:, None] ** 2 + 1e-8))
    gx = torch.exp(-(x[None, :] ** 2) / (2.0 * sigma_x[:, None] ** 2 + 1e-8))
    gz = gz / (gz.sum(dim=1, keepdim=True) + 1e-8)
    gx = gx / (gx.sum(dim=1, keepdim=True) + 1e-8)

    k = gz[:, :, None] * gx[:, None, :]  # B, kz, kx
    return k[:, None]                    # B, 1, kz, kx


def physical_forward(pred, meta):
    """
    用与 dataset.py 完全一致的前向模型重新退化预测结果：
        deg = A(z) * (PSF_z * PSF_x * pred)
    这是"物理一致性"唯一正确的写法；旧版漏掉了 A(z)，也用了错误的 sigma。
    """
    sigma_z = meta[:, 0].to(device=pred.device, dtype=pred.dtype)
    sigma_x = meta[:, 1].to(device=pred.device, dtype=pred.dtype)
    att_exp = meta[:, 2].to(device=pred.device, dtype=pred.dtype)

    k = make_psf_kernel(sigma_z, sigma_x)
    deg = F.conv2d(pred, k, padding=(k.shape[2] // 2, k.shape[3] // 2), groups=pred.shape[0])

    z = torch.linspace(0.0, 1.0, pred.shape[2], device=pred.device, dtype=pred.dtype)
    z = z[None, None, :, None]
    return deg * torch.exp(-att_exp[:, None, None, None] * z)


def total_loss(
    pred,
    target,
    raw_input=None,
    clean_deg=None,
    meta=None,
    weight=None,
    lam_mean=0.05,
    lam_grad=0.20,
    lam_laplace=0.05,
    lam_physics=1.0,
    lam_bg=0.5,
    lam_tv=0.01,
):
    # 1. 信号加权 Charbonnier：背景不再主导损失。
    #    v2 去掉了 target 的上限 clip，最亮像素可能到 2~4；权重必须封顶，
    #    否则单个镜面反射像素会主导整个 batch 的梯度。
    w = 1.0 + 4.0 * torch.clamp(target, 0.0, 1.0)
    if weight is not None:
        # 尺度加权：对 PAM 看不到的细结构降权，避免逼网络脑补
        w = w * weight
    loss_spatial = ((w * torch.sqrt((pred - target) ** 2 + 1e-3)).sum()
                    / (w.sum() + 1e-6))

    # 2. 能量/均值约束
    loss_mean = torch.abs(pred.mean() - target.mean())

    # 3. 一阶梯度（0.20，不再是 1.2）
    gp = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_grad = torch.mean(torch.sqrt((gp - gt) ** 2 + 1e-3))

    # 4. 二阶拉普拉斯
    lp = pred[:, :, 2:, :] - 2.0 * pred[:, :, 1:-1, :] + pred[:, :, :-2, :]
    lt = target[:, :, 2:, :] - 2.0 * target[:, :, 1:-1, :] + target[:, :, :-2, :]
    loss_laplace = torch.mean(torch.sqrt((lp - lt) ** 2 + 1e-3))

    # 5. 物理退化一致性（与 dataset 的前向模型严格一致）
    loss_physics = pred.new_zeros(())
    if clean_deg is not None and meta is not None:
        deg = physical_forward(pred, meta)
        loss_physics = F.l1_loss(deg, clean_deg)

    # 6. 空白区惩罚：输入无信号、目标无信号的地方，输出必须接近 0
    loss_bg = pred.new_zeros(())
    if raw_input is not None:
        bg_mask = ((target < 0.01) & (raw_input < 0.02)).to(pred.dtype)
        loss_bg = (pred * bg_mask).sum() / (bg_mask.sum() + 1e-6)

    # 7. TV（沿 Z 的轻度平滑）
    loss_tv = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))

    return (
        loss_spatial
        + lam_mean * loss_mean
        + lam_grad * loss_grad
        + lam_laplace * loss_laplace
        + lam_physics * loss_physics
        + lam_bg * loss_bg
        + lam_tv * loss_tv
    )
