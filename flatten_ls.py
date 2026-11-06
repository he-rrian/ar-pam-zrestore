"""在 data_reader 上加"按表面展平"的取片方式：每列按自己的 pia 平移，使表面固定在第 SURF_ROW 行。"""
import numpy as np

import config as C
import surface as S

SURF_ROW = 60


def flatten_cols(img, pia=None, surf_row=SURF_ROW):
    """把每列的 pia 对齐到 surf_row；返回展平后的图与每列位移。"""
    a = np.asarray(img, np.float32)
    if pia is None:
        pia = S.pia_from_ls(a)
    pia = np.asarray(pia, np.float32)
    ok = np.isfinite(pia)
    pia = np.where(ok, pia, surf_row).astype(np.int32)
    shift = surf_row - pia
    idx = np.arange(a.shape[0])[:, None] - shift[None, :]
    idx = np.clip(idx, 0, a.shape[0] - 1)
    return np.take_along_axis(a, idx, axis=0), shift


def unflatten_cols(flat, shift):
    """把展平的图还原到原坐标。"""
    a = np.asarray(flat, np.float32)
    idx = np.arange(a.shape[0])[:, None] + np.asarray(shift)[None, :]
    idx = np.clip(idx, 0, a.shape[0] - 1)
    return np.take_along_axis(a, idx, axis=0)
