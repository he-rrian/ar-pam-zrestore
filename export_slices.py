import numpy as np, os, config as C
e = np.load(os.path.join(C.WORK_DIR, "mouse4_full_enhanced.npy"), mmap_mode="r")
r = np.load(os.path.join(C.WORK_DIR, "4_pam_full_scaled.npy"), mmap_mode="r")
idx = [200, 450, 700, 900]
np.save(os.path.join(C.WORK_DIR, "enh_slices.npy"), np.stack([np.asarray(e[i]) for i in idx]))
np.save(os.path.join(C.WORK_DIR, "raw_slices.npy"), np.stack([np.asarray(r[i]) for i in idx]))
print("saved", len(idx), "slices", e.shape[1:])
