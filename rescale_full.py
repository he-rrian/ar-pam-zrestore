import os
import numpy as np
import config as C
K = 1.2302672451986854
arr = np.load(os.path.join(C.WORK_DIR, "4_pam_full.npy"), mmap_mode="r")
out = np.empty(arr.shape, np.float32)
for i in range(arr.shape[0]):
    out[i] = np.asarray(arr[i]) * K
np.save(os.path.join(C.WORK_DIR, "4_pam_full_scaled.npy"), out)
print("saved scaled", out.shape, float(out.mean()))
