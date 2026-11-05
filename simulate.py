"""物理退化模拟（v2）：y = A(z) * (PSF_x * PSF_z * x) + speckle + noise。

与 v1 的差别：
  * 不再对 pseudo 做 np.clip(..., 1.0)。伪数据的亮峰被削平，等于提前教网络
    "高光就是平顶"；真实数据里最亮的脑表面回波同样会被削平。
  * 衰减可以选择"相对脑表面"（ATT_RELATIVE_TO_SURFACE）。
  * 提供 degrade_only()，保证训练时的物理一致性损失和这里用的是同一个算子。
"""

import numpy as np
from scipy.signal import hilbert, fftconvolve
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import config as C


def attenuation(z_norm, att_exp, surface_norm=None):
    """A(z) = exp(-att * depth)，depth 可以是绝对 z 或相对表面的 z。"""
    if surface_norm is None:
        depth = z_norm
    else:
        depth = np.maximum(z_norm - surface_norm, 0.0)
    return np.exp(-float(att_exp) * depth)


def _gabor(sigma_px, f0_cyc_per_px, k=None):
    """
    换能器冲激响应的一阶模型：高斯包络 × 载波。
    为什么不是高斯低通：超声是**带通**的，轴向 PSF 来自脉冲响应而不是低通核；
    用带通脉冲还能自然产生"相邻两个散射体在脉冲宽度内相互干涉"的效果，
    这正是要分辨 10~15 μm 微层时最需要仿真的部分。
    """
    if k is None:
        k = int(np.ceil(10 * sigma_px)) | 1
    t = np.arange(k, dtype=np.float32) - k // 2
    g = np.exp(-t ** 2 / (2.0 * sigma_px ** 2)) * np.cos(2 * np.pi * f0_cyc_per_px * t)
    # 归一化到“单点脉冲的包络峰值为 1”，退化不改变整体幅度
    env = np.abs(hilbert(g))
    return g / (env.max() + 1e-12)


def degrade_rf(ls_xz, sigma_z=None, sigma_x=None, att_exp=None, surface=None,
               f0=C.RF_F0_CYC_PER_SAMPLE):
    """
    RF 域退化：反射率 -> 带通脉冲卷积 -> 深度衰减 -> Hilbert 包络。
    返回 (env, target, clean_linear)：
      env          : 非线性包络（含散射体干涉），作为网络输入的基础
      target       : 横向匹配后的光片（监督）
      clean_linear : A(z)·G(σz)*target，线性退化，供物理一致性损失（与 model.py 一致）
    """
    from scipy.ndimage import gaussian_filter1d as _g1
    sigma_z = C.PSF_SIGMA_Z if sigma_z is None else float(sigma_z)
    sigma_x = C.PSF_SIGMA_X if sigma_x is None else float(sigma_x)
    att_exp = C.ATT_EXP if att_exp is None else float(att_exp)

    target = gaussian_filter(ls_xz, sigma=(0.0, sigma_x))
    z_norm = np.linspace(0.0, 1.0, target.shape[0])[:, None]
    att = attenuation(z_norm, att_exp, surface)

    # RF 域（1 RF sample = 1.43 个 2.1 μm 像素）
    f0_px = float(f0) / C.RF_SAMPLES_PER_PX
    g = _gabor(sigma_z * C.RF_SIGMA_CORR, f0_px)   # sigma 用 457 网格的像素单位
    rf = fftconvolve(target, g[:, None], mode="same", axes=0).astype(np.float32)
    rf *= att
    env = np.abs(hilbert(rf, axis=0)).astype(np.float32) * C.RF_GAIN_CAL

    clean = (_g1(target, sigma_z, axis=0) * att).astype(np.float32)
    return env, target.astype(np.float32), clean


def degrade_only(ls_xz, sigma_z=None, sigma_x=None, att_exp=None, surface=None):
    """
    只做确定性退化（横向 PSF -> 轴向 PSF -> 深度衰减），不做噪声。
    返回 clean；target（= 横向匹配后的光片）另行返回。
    """
    sigma_z = C.PSF_SIGMA_Z if sigma_z is None else float(sigma_z)
    sigma_x = C.PSF_SIGMA_X if sigma_x is None else float(sigma_x)
    att_exp = C.ATT_EXP if att_exp is None else float(att_exp)

    target = gaussian_filter(ls_xz, sigma=(0.0, sigma_x))
    blurred = gaussian_filter1d(target, sigma=sigma_z, axis=0)

    z_norm = np.linspace(0.0, 1.0, target.shape[0])[:, None]
    if surface is not None:
        surf = np.asarray(surface, dtype=np.float32)
        if surf.ndim == 1:
            surf = surf[None, :]
        att = attenuation(z_norm, att_exp, surf)
    else:
        att = attenuation(z_norm, att_exp, None)

    return (blurred * att).astype(np.float32), target.astype(np.float32)


def generate_pseudo_pam(ls_xz, rng=None, sigma_z=None, sigma_x=None, att_exp=None,
                        speckle_scale=None, noise_sigma=None, noise_blur=None,
                        surface=None, gain=1.0, mode=None):
    """
    返回 (pseudo, target, clean)：
      target : 横向匹配到 AR-PAM 的光片（监督目标）
      clean  : A(z)*G(σz)*target（与 model.physical_forward 一致的线性退化）
      pseudo : mode="rf" 时用带通脉冲 + 包络（含干涉），否则用高斯包络
    """
    if rng is None:
        rng = np.random.default_rng()
    mode = C.SIM_MODE if mode is None else mode
    sigma_z = C.PSF_SIGMA_Z if sigma_z is None else float(sigma_z)
    sigma_x = C.PSF_SIGMA_X if sigma_x is None else float(sigma_x)
    att_exp = C.ATT_EXP if att_exp is None else float(att_exp)
    speckle_scale = C.SPECKLE_SCALE if speckle_scale is None else float(speckle_scale)
    noise_sigma = C.NOISE_SIGMA if noise_sigma is None else float(noise_sigma)
    noise_blur = C.NOISE_BLUR if noise_blur is None else float(noise_blur)

    if mode == "rf":
        clean, target, clean_lin = degrade_rf(ls_xz, sigma_z, sigma_x, att_exp, surface)
        clean_for_loss = clean_lin
    else:
        clean, target = degrade_only(ls_xz, sigma_z, sigma_x, att_exp, surface)
        clean_for_loss = clean

    speckle = rng.rayleigh(speckle_scale, size=clean.shape).astype(np.float32)
    gauss = rng.normal(0.0, noise_sigma, size=clean.shape).astype(np.float32)
    if noise_blur and noise_blur > 0:
        speckle = gaussian_filter1d(speckle, noise_blur, axis=0) * 1.3
        gauss = gaussian_filter1d(gauss, noise_blur, axis=0) * 1.3

    base = clean * float(gain)
    pseudo = np.maximum(base + base * speckle + gauss, 0.0).astype(np.float32)
    return pseudo, target, clean_for_loss
