"""用与正式缓存完全相同的预处理，构建全深度(0:500 -> 714px = 1500um)缓存。"""
import config as C

FULL_PX = 714          # 500 samples * 3.0 um / 2.1 um = 714.3
C.TARGET_Z = FULL_PX

import pam_io
pam_io.C.TARGET_Z = FULL_PX

out = r"D:\tmp\work\4_pam_full.npy"
pam_io.build_cache(C.PAM_M4_TXT_DIR, out, z_start=0, z_end=500, force=True)
print("done", out)
