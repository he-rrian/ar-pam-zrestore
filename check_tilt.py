"""量化光片脑表面的倾斜：在全网格上拟合 pia = a + b*Y + c*X（层数）。"""
import numpy as np

import config as C
import surface as S

for m, gridp in [("4", C.IMS_M4_NPY), ("1", C.IMS_M1_NPY)]:
    g = np.load(gridp, mmap_mode="r")          # (457, 4700, 1000)
    ys = np.arange(0, 4700, 40)
    xs = np.arange(0, 1000, 25)
    pts = []
    for y in ys:
        pia = S.pia_from_ls(np.asarray(g[:, y, :], np.float32))
        for x in xs:
            v = pia[x]
            if np.isfinite(v) and 3 < v < 450:
                pts.append((y, x, v))
    P = np.array(pts)
    A = np.c_[np.ones(len(P)), P[:, 0], P[:, 1]]
    coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    r = P[:, 2] - A @ coef
    print(f"mouse{m}: n={len(P)}  pia ≈ {coef[0]:.0f} + {coef[1]*1000:.1f}*Ymm + {coef[2]*1000:.1f}*Xmm  (层数)")
    print(f"         斜率: Y {coef[1]*C.LS_PITCH_UM*1000:.0f} um/mm, X {coef[2]*C.LS_PITCH_UM*1000:.0f} um/mm")
    print(f"         总落差: Y 方向 {(P[:,0].max()-P[:,0].min())*coef[1]:.0f} 层 = {(P[:,0].max()-P[:,0].min())*coef[1]*C.LS_PITCH_UM/1000:.2f} mm；"
          f"X 方向 {(P[:,1].max()-P[:,1].min())*coef[2]:.0f} 层")
    print(f"         残差 std {r.std():.1f} 层 = {r.std()*C.LS_PITCH_UM:.0f} um")
