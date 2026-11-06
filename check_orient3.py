"""用"表面深度 vs Y"的斜率判断光片与声光的 Y 轴是否同向（斜率反号即翻转）。"""
import numpy as np

import config as C
import pam_io
import surface as S

for m, cache, gridp in [("4", C.CACHE_M4_PATH, C.IMS_M4_NPY), ("1", C.CACHE_M1_PATH, C.IMS_M1_NPY)]:
    raw, _ = pam_io.load_cache(cache)
    grid = np.load(gridp, mmap_mode="r")
    NY = raw.shape[0]

    # 声光：每 5 帧取一帧，求表面层
    ys, ps = [], []
    for i in range(0, NY, 5):
        pia = S.pia_from_ls(np.asarray(raw[i], np.float32))
        v = np.nanmedian(pia)
        if np.isfinite(v) and v > 3:
            ys.append(i); ps.append(v)
    ys = np.array(ys); ps = np.array(ps)
    # 光片：每 20 个 Y 取一列，求表面层
    lys, lss = [], []
    for y in range(0, 4700, 20):
        pia = S.pia_from_ls(np.asarray(grid[:, y, :], np.float32))
        v = np.nanmedian(pia)
        if np.isfinite(v) and v > 3:
            lys.append(y); lss.append(v)
    lys = np.array(lys); lss = np.array(lss)

    sp = np.polyfit(ys, ps, 1)[0] if len(ys) > 5 else np.nan
    sl = np.polyfit(lys, lss, 1)[0] if len(lys) > 5 else np.nan
    # 统一到"每 mm 变化层数"：声光 10 μm/帧，光片 2.1 μm/行
    sp_mm = sp * C.PAM_FRAME_PITCH_UM / 1000   # 层/mm？这里声光是 px/帧 -> 换成 px/mm 需 *1000/10
    sp_mm = sp * 1000 / C.PAM_FRAME_PITCH_UM
    sl_mm = sl * 1000 / C.LS_PITCH_UM
    print(f"mouse{m}: 声光表面斜率 {sp:.4f} 层/帧 = {sp_mm:+.1f} 层/mm (n={len(ys)})")
    print(f"          光片表面斜率 {sl:.4f} 层/行 = {sl_mm:+.1f} 层/mm (n={len(lys)})")
    print(f"          -> {'同向' if sp_mm * sl_mm > 0 else '反向（需要翻转 Y）'}")
