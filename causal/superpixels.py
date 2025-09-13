import numpy as np
import torch
from skimage.segmentation import slic

from causal.utils import img_to_numpy


def get_superpixels(img_tensor, n_segments, return_num_segs=False, **kwargs):
    """:param n_segments is only approximate"""
    img = img_to_numpy(img_tensor)
    seg = slic(img, n_segments=n_segments, **kwargs)
    seg_tensor = torch.from_numpy(seg).to(img_tensor.device)

    if return_num_segs:
        num_segs = len(torch.unique(seg_tensor))
        return seg_tensor, num_segs

    return seg_tensor


def get_block_seg(img_tensor, y_num, x_num, return_size=False, return_num_segs=False):
    """:param y_num, x_num are num blocks over height and width"""
    img = img_to_numpy(img_tensor)  # H*W*C

    h, w = img.shape[:2]  # num_rows, num_cols
    seg = np.zeros((h, w), dtype=np.int_)

    h_div, y_size = np.linspace(0, h, num=y_num + 1, dtype=np.int_, retstep=True)
    w_div, x_size = np.linspace(0, w, num=x_num + 1, dtype=np.int_, retstep=True)

    count = 0  # lime recommendation
    for y_start, y_end in zip(h_div[:-1], h_div[1:]):
        for x_start, x_end in zip(w_div[:-1], w_div[1:]):
            seg[y_start: y_end, x_start:x_end] = count
            count += 1

    assert (count == y_num * x_num)

    seg_tensor = torch.from_numpy(seg).to(img_tensor.device)

    if (h % y_num == 0) and (w % x_num == 0):
        block_sizes = torch.bincount(torch.flatten(seg_tensor))[1:]
        actual_size = int(y_size * x_size)
        assert torch.all(block_sizes == actual_size)

        if return_size:
            return seg_tensor, actual_size

    if return_num_segs:
        return seg_tensor, x_num * y_num

    return seg_tensor


def batched_from_interp_rep_transform(batched_sample, original_inputs, feature_mask, **kwargs):
    batched_masks = torch.index_select(batched_sample, 1, feature_mask.flatten())
    batched_masks = batched_masks.reshape(batched_sample.shape[0], 1, *original_inputs.shape[-2:])

    return batched_masks * original_inputs
