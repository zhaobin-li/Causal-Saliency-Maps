from itertools import product

import numpy as np
import torch
from skimage.transform import resize
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def original_BT_masks(n_masks=1, mask_shape=(100, 100), ls=(1 / 10, 1 / 10), target_shape=(224, 224), mu=-100, sigma=100):
    """
    ls: length scale determines the blob shape relative to mask_shape
    mu: determines amount of 0s and 1s. more negative means more zeros
    sigma: determines rate of change between 0s and 1s. larger means quicker change
    """

    # mask_shape don't go over (100, 100)
    x = np.linspace(0, 1, mask_shape[0])
    y = np.linspace(0, 1, mask_shape[1])
    xv, yv = np.meshgrid(x, y)
    xy = np.zeros([np.prod(mask_shape), 2])
    xy[:, 0] = xv.flatten()
    xy[:, 1] = yv.flatten()

    kernel = RBF(length_scale=ls)
    K = kernel(xy)
    chol_K = np.linalg.cholesky(K + np.eye(np.prod(mask_shape)) * 10 ** (-6)) # 1e-6 added to stabilise the cholesky

    s = np.random.randn(np.prod(mask_shape), n_masks) * sigma
    gp_samps = np.matmul(chol_K, s) + mu
    masks = sigmoid(gp_samps)

    # reshape masks to resized target image shape
    reshaped_masks = []

    for i in range(n_masks):
        mask = resize(masks[:, i].reshape(mask_shape), target_shape)
        reshaped_masks.append(mask)

    reshaped_masks = np.asarray(reshaped_masks)
    return reshaped_masks


def elementwise_product(batched_sample, original_input, batched_sample_channels, **kwargs):
    return batched_sample.reshape(-1, batched_sample_channels, *original_input.shape[2:]) * original_input  # broadcast


def original_rise_masks(n_samples, num_per_side, prob=0.5, input_size=(224, 224)):
    cell_size = np.ceil(np.array(input_size) / num_per_side)
    up_size = (num_per_side + 1) * cell_size

    grid = np.random.rand(n_samples, num_per_side, num_per_side) < prob
    grid = grid.astype('float32')

    masks = np.empty((n_samples, *input_size))

    for i in range(n_samples):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]

    masks = masks.reshape(-1, 1, *input_size)
    masks = torch.from_numpy(masks).float()

    return masks


def balanced_trials_gen(_original_input, n_samples, num_interp_features, seed=None, **kwargs):
    """generator :param _original_input and :rtype required as stated in documentation"""

    # create balanced matrix
    balanced_trials = np.zeros([n_samples, num_interp_features], dtype=np.int_)
    balanced_trials[int(n_samples / 2):] = 1

    # permute assignment per feature
    rng = np.random.default_rng(seed)
    balanced_trials = rng.permuted(balanced_trials, axis=0)

    balanced_trials = torch.from_numpy(balanced_trials).to(_original_input.device)

    # iterate per sample
    for sample in balanced_trials:
        yield sample.unsqueeze(0)


def balanced_trials_gen_batched(_original_input, n_samples, num_interp_features, seed=None, **kwargs):
    """generator :param _original_input and :rtype required as stated in documentation"""

    # create balanced matrix
    balanced_trials = np.zeros([n_samples, num_interp_features], dtype=np.int_)
    balanced_trials[int(n_samples / 2):] = 1

    # permute assignment per feature
    rng = np.random.default_rng(seed)
    balanced_trials = rng.permuted(balanced_trials, axis=0)

    return torch.from_numpy(balanced_trials).to(_original_input.device)


def permutation_gen(_original_input, n_samples, num_interp_features, **kwargs):
    """generator :param _original_input and :rtype required as stated in documentation"""
    permutations = torch.tensor(list(product([0, 1], repeat=num_interp_features))).to(_original_input.device)

    for sample in permutations:
        yield sample.unsqueeze(0)


def permutation_gen_batched(_original_input, n_samples, num_interp_features, **kwargs):
    """generator :param _original_input and :rtype required as stated in documentation"""
    return torch.tensor(list(product([0, 1], repeat=num_interp_features))).to(_original_input.device)


def bernoulli_trials_gen(_original_input, n_samples, num_interp_features, p=0.5, seed=None, **kwargs):
    """generator :param _original_input and :rtype required as stated in documentation"""

    rng = np.random.default_rng(seed)

    bernoulli_trials = rng.binomial(n=1, p=p, size=(n_samples, num_interp_features))
    bernoulli_trials = torch.from_numpy(bernoulli_trials).to(_original_input.device)

    # iterate per sample
    for sample in bernoulli_trials:
        yield sample.unsqueeze(0)


def bernoulli_trials_gen_batched(_original_input, n_samples, num_interp_features, p=0.5, seed=None, **kwargs):
    """generator :param _original_input and :rtype required as stated in documentation"""

    rng = np.random.default_rng(seed)

    bernoulli_trials = rng.binomial(n=1, p=p, size=(n_samples, num_interp_features))
    return torch.from_numpy(bernoulli_trials).to(_original_input.device)
