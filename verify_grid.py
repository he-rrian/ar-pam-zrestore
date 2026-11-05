import numpy as np, config as C
from data_reader import compute_ls_stats, block_mean_x, LightsheetReader, load_or_compute_stats

for m, raw_p, gp in [("4", C.IMS_M4_RAW, "work/4_ls_grid.npy"), ("1", C.IMS_M1_RAW, "work/1_ls_grid.npy")]:
    st = compute_ls_stats(raw_p)
    raw = np.load(raw_p, mmap_mode="r"); grid = np.load(gp, mmap_mode="r")
    worst = 0.0; worst_rel = 0.0; mincorr = 1.0
    for y in [300, 1200, 2400, 3600, 4500]:
        ref = np.asarray(raw[0:C.TARGET_Z, y, :], np.float32)
        sig = np.maximum(ref - st["bg"], 0.0)
        sig = np.maximum(sig - st["col_profile"][None, :], 0.0)
        refn = block_mean_x(sig, C.TARGET_X) / st["gain"]
        g = np.asarray(grid[:, y, :], np.float32)
        d = np.abs(g - refn)
        worst = max(worst, float(d.max()))
        worst_rel = max(worst_rel, float(d.max() / (refn.max() + 1e-9)))
        mincorr = min(mincorr, float(np.corrcoef(g.ravel(), refn.ravel())[0, 1]))
    r1 = LightsheetReader(raw_p, stats=st)
    r2 = LightsheetReader(gp, stats=load_or_compute_stats(gp))
    a = r1.get_xz(2350 + C.LS_Y_START); b = r2.get_xz(2350 + C.LS_Y_START)
    print(f"mouse{m}: grid {grid.shape} {grid.dtype} | max|diff| {worst:.4f} (rel {worst_rel:.4f}) "
          f"min corr {mincorr:.5f} | get_xz max|diff| {np.abs(a-b).max():.4f}")
