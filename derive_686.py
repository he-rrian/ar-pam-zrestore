import os
import numpy as np
import config as C
K = 1.2302672451986854
a = np.load(os.path.join(C.WORK_DIR, "4_pam_full.npy"), mmap_mode="r")   # (998,714,1000)
out = np.empty((a.shape[0], 686, a.shape[2]), np.float32)
for i in range(a.shape[0]):
    out[i] = np.asarray(a[i])[28:714] * K
np.save(os.path.join(C.WORK_DIR, "4_pam_full686s.npy"), out)
print("saved", out.shape, float(out.mean()))
