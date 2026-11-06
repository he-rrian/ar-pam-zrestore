import numpy as np, os, config as C, pam_io
raw, _ = pam_io.load_cache(os.path.join(C.WORK_DIR, "1_pam_v2.npy"))
enh = np.load(os.path.join(C.WORK_DIR, "mouse1_enhanced_v2.npy"), mmap_mode="r")
idx = [150, 400, 650, 880]
np.save(os.path.join(C.WORK_DIR, "pam_slices_1.npy"), np.stack([np.asarray(raw[i]) for i in idx]))
np.save(os.path.join(C.WORK_DIR, "enh_slices_1.npy"), np.stack([np.asarray(enh[i]) for i in idx]))
print("saved", len(idx), raw.shape[1:])
