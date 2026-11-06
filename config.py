"""
全局配置（v2，2024-05 数据实测标定）。

v1 -> v2 的变化（都有实测依据，见 REVIEW_v2.md）：

  1. 奇偶 = 双向往返扫描：奇数帧在 X（A-line）方向是镜像的。必须
     `raw_rf[::-1, :]` 翻转后再堆叠，否则 Y 向每隔一帧左右颠倒，
     2.5D 上下文一半是反的。翻转后：1 帧 = 10 μm，1000 帧 = 10 mm。
     （上一版误判成"两套对比通道"，见 REVIEW_v2.md §1 的更正。）

  2. 所有"像素"参数改成"微米"参数，再换算成像素。v1 里 4.7x / 10 um
     这类常数散落在各个脚本里，是上一轮 Y 映射错误的根源。

  3. 归一化一律"只减背景、整卷统一 gain、不设上限 clip"。
     v1 的 clip(...,1.0) 会把脑表面回波和几条最亮血管削成平顶——
     这就是"txt 里没削顶、处理完削顶"的来源。

  4. 深度窗口 180:500 -> 20:340（320 samples = 960 um）。
     实测皮层表面回波前沿在 RF sample 31~190（mouse1 中位数 55，
     mouse4 中位数 122），v1 的 60:380 对 mouse1 只剩 10 个 sample 余量，
     左/右侧穹顶前沿仍被切；20:340 对两鼠都留 >=10 sample 余量。
"""

import os

# =============================================================================
# 0. 路径
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")        # 原始数据，只读
WORK_DIR = os.path.join(BASE_DIR, "work")        # 派生数据（缓存/增强结果）

# A 方案：686 层 = 1440 um，与 PAM 20:500 严格对齐（必须在 _ls_path 之前定义）
FULL_DEPTH = True
QC_DIR = os.path.join(BASE_DIR, "qc")            # 体检报告图
for _d in (WORK_DIR, QC_DIR):
    os.makedirs(_d, exist_ok=True)

IMS_M4_RAW = os.path.join(DATA_DIR, "4_subset.npy")   # 原始 19 GB（ls_grid.py 的输入）
IMS_M1_RAW = os.path.join(DATA_DIR, "1_subset.npy")


def _ls_path(mouse):
    """有"训练即用"网格就用网格（4.3 GB），否则退回原始 npy（19 GB）。"""
    g_full = os.path.join(WORK_DIR, f"{mouse}_ls_grid_full.npy")
    if FULL_DEPTH and os.path.exists(g_full):
        return g_full
    g = os.path.join(WORK_DIR, f"{mouse}_ls_grid.npy")
    return g if os.path.exists(g) else os.path.join(DATA_DIR, f"{mouse}_subset.npy")


IMS_M4_NPY = _ls_path("4")
IMS_M1_NPY = _ls_path("1")

PAM_M1_TXT_DIR = os.path.join(DATA_DIR, "1")
PAM_M4_TXT_DIR = os.path.join(DATA_DIR, "4")

# 声光缓存（v2 格式：float32 (N,457,1000) + 同名 .json 元数据）
CACHE_M1_PATH = os.path.join(WORK_DIR, "1_pam_v2.npy")
CACHE_M4_PATH = os.path.join(WORK_DIR, "4_pam_full686s.npy" if FULL_DEPTH else "4_pam_v2.npy")

# v1 旧缓存（逐切片归一化 + 旧窗口 + 未拆奇偶），不能再使用
LEGACY_CACHE_FILES = [
    os.path.join(DATA_DIR, "1_pam_cache.npy"),
    os.path.join(DATA_DIR, "4_pam_cache.npy"),
]

# =============================================================================
# 1. 物理网格（单位：um）
# =============================================================================
LS_PITCH_UM = 2.1                 # 光片各向同性体素
LS_Z_LAYERS = 457                 # npy 的 Z 层数  -> 960 um
LS_Y_ROWS = 4700                  # npy 的 Y 行数  -> 9870 um
LS_X_PIX = 4766                   # npy 的 X 宽度  -> 10010 um
LS_Y_START = 200                  # npy 的 Y=0 对应原始 ims 的 Y=200

PAM_RF_PITCH_UM = 3.0             # RF 采样（声速 1540 m/s, 2x 过采样）-> 3 um/sample
PAM_RF_SAMPLES = 500              # 每个 A-line 的 RF 采样数 -> 1500 um
PAM_ALINES = 1000                 # 每个 txt 的 A-line 数
PAM_BEAM_PITCH_UM = 10.0          # A-line 间距（X）
PAM_ACQ_FRAME_PITCH_UM = 10.0     # 采集时相邻 txt 的 Y 步进

# 奇偶两套的物理含义（实测，见 REVIEW_v2.md §1）：
#   双向往返光栅扫描：偶数帧 X 正扫、奇数帧 X 反扫（镜像）。
#   翻转奇数帧的 A-line 轴后，相邻帧相关 0.17 -> 0.93、亮结构重合 0.2 -> 0.95。
PAM_FLIP_ODD_X = True
PAM_FRAME_PITCH_UM = PAM_ACQ_FRAME_PITCH_UM      # 10 um，1000 帧 = 10 mm

# 统一网络网格
TARGET_Z = 686 if FULL_DEPTH else 457
TARGET_X = 1000                   # 10000 um / 10 um（X 按 10 um 分箱）
LS_X_BIN_UM = LS_PITCH_UM * LS_X_PIX / TARGET_X   # = 10.007 um

# =============================================================================
# 2. 声光深度窗口
# =============================================================================
PAM_Z_START = 20                  # RF sample；跳过 0~12 的触发/振铃伪影
PAM_Z_END = 500 if FULL_DEPTH else PAM_Z_START + 320
PAM_Z_WINDOW = PAM_Z_END - PAM_Z_START

# 奇偶处理结果：始终保留全部 1000 帧（奇数帧翻转后与偶数帧同向）
# 若哪天数据换成"两套真实对比通道"，把下面的开关设为 False 并改用
# pam_io.build_cache(..., parity=...) 的旧分支。

# =============================================================================
# 3. 光片标定（对 data/*_subset.npy 实测）
# =============================================================================
# uint16，背景中位数 415~422；最亮像素 3216；未饱和（<65535）
LS_BG_PERCENTILE = 50.0
LS_GAIN_PERCENTILE = 99.9
LS_COL_PERCENTILE = 10.0          # 列平场（照明条纹）
LS_COL_SMOOTH = 5.0

# =============================================================================
# 4. 声光标定（对 data/{1,4}/*.txt 实测，见 calibrate_psf.py / qc_report.py）
# =============================================================================
PAM_BG_PERCENTILE = 50.0          # 每帧背景（离焦雾）
PAM_GAIN_PERCENTILE = 99.9

# 归一化：只减背景，整卷统一 gain，不设上限
GAIN_HEADROOM = 1.3               # gain = p99.9 * 1.3  ->  p99.9 -> 0.77

# 轴向 PSF（457 网格，2.1 um/px）：实测 304 个孤立峰，sigma 中位数 2.43 RF
# samples = 7.3 um = 3.47 px（p10 2.5 px, p90 5.4 px）。
# 注意：这是"有效 PSF"，已包含血管自身直径，真实点源只会更窄。
PSF_SIGMA_Z = 3.3
PSF_SIGMA_Z_MIN = 2.3
PSF_SIGMA_Z_MAX = 4.9

# 横向 PSF：AR-PAM 横向分辨率远低于 10 um 采样；光片 4766->1000 分箱后
# 约为 10 um 有效分辨率。这个值只影响 X，不影响要恢复的 Z。
PSF_SIGMA_X = 1.0
PSF_SIGMA_X_MIN = 0.4
PSF_SIGMA_X_MAX = 1.8

# 深度衰减：实测真实 PAM 在 z=440 px 处只剩 0.27，对应 exp(-1.3 z/Z)
ATT_EXP = 1.3
ATT_EXP_MIN = 0.9
ATT_EXP_MAX = 1.7
# 是否把衰减起点放在每帧的脑表面（更物理，但 v1 未验证；默认关闭）
ATT_RELATIVE_TO_SURFACE = False

# 噪声：真实包络噪声被换能器带宽平滑过，不是白噪声
SPECKLE_SCALE = 0.05
NOISE_SIGMA = 0.008
NOISE_BLUR = 1.0
SPECKLE_SCALE_MIN = 0.02
SPECKLE_SCALE_MAX = 0.09
NOISE_SIGMA_MIN = 0.004
NOISE_SIGMA_MAX = 0.018

# 整卷 gain 归一化之后，两卷的真实声光仍可能差 1.5 倍以上
# （实测 p99：mouse1 0.41，mouse4 0.27）。训练时随机化输入的整体增益，
# 避免网络把"整体亮度"当作结构信息。只作用在输入上，不改变物理损失。
PSEUDO_GAIN_MIN = 0.70
PSEUDO_GAIN_MAX = 1.40

# ---------------- 伪数据的退化模型 ----------------
# "rf"  : RF 域带通脉冲（高斯包络 × 载波）+ 深度衰减 -> Hilbert 包络。
#         比高斯低通多了"脉冲宽度内相邻散射体干涉"和真实的旁瓣结构。
#         注意：Gabor 的包络比同 sigma 的高斯宽 ~1.25 倍、幅度高 ~5.8 倍，
#         所以下面两个标定常数必须一起用，否则 PSF 和亮度都会偏。
# "env" : 直接对包络做高斯模糊（已用 304 个孤立峰标定，作为基线/默认）。
SIM_MODE = "env"
# 换能器中心频率（cycles / RF sample；500 sample = 1500 μm 往返）
# 由轴向 PSF 的 FWHM≈5.7 sample 反推：λ≈8~10 sample -> f0≈0.10~0.12
RF_F0_CYC_PER_SAMPLE = 0.10
RF_SIGMA_CORR = 0.82       # σ_gabor = σ_target * 该系数（校准到包络 FWHM 一致）
RF_GAIN_CAL = 0.172        # env *= 该系数（校准到与线性退化同均值）
# 1 个 2.1 μm 像素 = 多少 RF sample（3 μm/sample）
RF_SAMPLES_PER_PX = PAM_RF_PITCH_UM / LS_PITCH_UM

# ---------------- 目标尺度加权（与 Gemini 对齐后的定稿）----------------
# 实测：PAM 沿 X 的血管半高宽中位 33 μm（p10 24），光片同网格 38 μm；
# 说明 AR-PAM 的横向有效尺度约 20~30 μm，比它更细的结构属于"看不到"。
# 做法不是删像素，而是对"小于 THIN_SCALE_PX 的结构"降监督权重：
#   权重 = 1 / (1 + strength * 细结构占比)，下限 floor。
# 这样输入（声光）与物理前向都不变，保留血管的 Z 向监督也不变。
TARGET_SCALE_WEIGHT = True
THIN_SCALE_PX = 2.0        # 2 px = 20 μm（约等于横向 PSF 下限）
THIN_WEIGHT_STRENGTH = 1.0
THIN_WEIGHT_FLOOR = 0.40

# 表面回波：实测 回波/血管p99 中位 0.59、p90 3.19，深 154±130 层、半高宽 7 层。
# 伪数据默认没有回波；若增强结果在表面出现亮弧伪影，把 SURFACE_ECHO 打开。
SURFACE_ECHO = True
ECHO_PROB = 0.30          # 每个样本按该概率叠表面回波（域随机化，而不是二选一）
ECHO_AMP = 0.6
ECHO_SIGMA = 1.5

# =============================================================================
# 5. 光片 Z 裁剪（重裁 .ims 时用；npy.py 读取这里）
# =============================================================================
# 原始 .ims：mouse4 (2048, 6272, 5504)；mouse1 (2048, 6272, 4864)
# 背侧表面位置：mouse4 在**高 Z**（1900~2000）-> 翻转；mouse1 在**低 Z**（~100）-> 不翻。
# 旧裁剪 1540:1997 上方只剩 51 层（107 μm），贴顶；改成 1590:2047 可多拿 51 层背景。
LS_CROP = {
    "4": {"z_start": 1590, "n_layers": 457, "flip_z": True},
    "1": {"z_start": 0,    "n_layers": 457, "flip_z": False},
}
# 分层评估用的"宽 slab"：留足表面起伏（约 300 层）+ 960 μm 皮层深度
LS_CROP_TALL_N = 700
# 表面距窗口上沿不足这么多层时，该 Y 不参与分层统计（方式 A）
LS_SURFACE_MARGIN = 8

# =============================================================================
# 6. 2.5D 上下文（单位：um，沿 Y）
# =============================================================================
# 1000 帧 @ 10 um，5 帧上下文 = ±20 um。
# 光片侧按同一物理距离取层：offset_um / 2.1 -> 整数层号（±10、±5）。
PAM_CONTEXT_UM = (-20.0, -10.0, 0.0, 10.0, 20.0)
CONTEXT_LS_LAYERS = tuple(int(round(o / LS_PITCH_UM)) for o in PAM_CONTEXT_UM)  # (-10,-5,0,5,10)

# =============================================================================
# 6. 训练 / 数据切分
# =============================================================================
TRAIN_Y_END = int(LS_Y_START + LS_Y_ROWS * 0.80)     # 3960
VAL_Y_START = TRAIN_Y_END + 50                       # 4010
VAL_Y_END = LS_Y_START + LS_Y_ROWS - 15              # 4885
TRAIN_Y_STRIDE = 3
VAL_Y_STRIDE = 5

BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 80
LR = 4e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# 损失权重
LAM_MEAN = 0.05
LAM_GRAD = 0.50
LAM_LAPLACE = 0.05
LAM_PHYSICS = 1.0
LAM_BG = 0.5
LAM_TV = 0.01
LAM_THIN = 0.0
# 第 3 轮：把 RL 反卷积结果作为第 6 个输入通道（物理先验）
USE_RL_PRIOR = True
# 第 7 轮：按脑表面展平后再造伪配对（表面固定，960um 窗口处处完整）
FLATTEN_LS = True
# 第 4 轮：L1/Charbonnier 的最优解偏向平滑，改用 L2 让峰值/振铃被对称惩罚
LOSS_L2 = True
RL_ITERS = 15           # 反幻觉：惩罚比 clean 更细的输出结构


def pam_frame_to_ls_row(frame_idx):
    """PAM（拆奇偶后）第 k 帧 -> 光片 npy 的 Y 行号（近似，仅用于评估对齐）。"""
    y_um = frame_idx * PAM_FRAME_PITCH_UM
    return int(round(y_um / LS_PITCH_UM))


def ls_row_to_pam_frame(ls_row):
    """光片 npy 的 Y 行号 -> 最近的 PAM 帧号（近似）。"""
    return int(round(ls_row * LS_PITCH_UM / PAM_FRAME_PITCH_UM))


def summary():
    lines = [
        "光片: %d x %d x %d @ %.1f um  ->  %.2f x %.2f x %.2f mm"
        % (LS_Z_LAYERS, LS_Y_ROWS, LS_X_PIX, LS_PITCH_UM,
           LS_X_PIX * LS_PITCH_UM / 1000, LS_Y_ROWS * LS_PITCH_UM / 1000,
           LS_Z_LAYERS * LS_PITCH_UM / 1000),
        "声光: %d A-line @ %.0f um / %d RF sample @ %.0f um"
        % (PAM_ALINES, PAM_BEAM_PITCH_UM, PAM_RF_SAMPLES, PAM_RF_PITCH_UM),
        "声光 Y: 每 %.0f um 一帧，奇数帧翻转后 %d um/帧 -> 998/1000 帧 = 10 mm"
        % (PAM_ACQ_FRAME_PITCH_UM, PAM_FRAME_PITCH_UM),
        "窗口: RF %d:%d (%d samples = %.0f um) -> %d px @ %.1f um"
        % (PAM_Z_START, PAM_Z_END, PAM_Z_WINDOW,
           PAM_Z_WINDOW * PAM_RF_PITCH_UM, TARGET_Z, LS_PITCH_UM),
        "2.5D: %s um -> 光片层偏移 %s"
        % (PAM_CONTEXT_UM, CONTEXT_LS_LAYERS),
        "PSF: sigma_z=%.1f px [%.1f, %.1f]  sigma_x=%.1f px  att=%.1f"
        % (PSF_SIGMA_Z, PSF_SIGMA_Z_MIN, PSF_SIGMA_Z_MAX, PSF_SIGMA_X, ATT_EXP),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
