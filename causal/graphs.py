import numpy as np
import torch
from captum.attr._utils import visualization as viz
from matplotlib import pyplot as plt

from causal.utils import img_to_numpy, unnormalize


def graph_auc_curve(pixels, scores, auc, ax=None, title=None):
    """Based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    pixels = pixels / max(pixels)

    lw = 2
    ax.plot(
        pixels,
        scores,
        color="darkorange",
        lw=lw,
        label="area = %0.2f" % auc,
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Image Fraction")
    ax.set_ylabel("Class probability")

    if title is not None:
        ax.set_title(title)

    ax.legend(loc="lower right")

    if ax is None:
        plt.show()
        return fig, ax


def graph_saliency_map(attrs, img, titles=None, auc_results=None, **kwargs):
    if isinstance(attrs, torch.Tensor):
        attrs = img_to_numpy(attrs)

    if isinstance(img, torch.Tensor):
        img = img_to_numpy(unnormalize(img))

    num_subplts = 2
    if auc_results:
        num_subplts += 1

    fig, ax = plt.subplots(1, num_subplts, figsize=(6 * num_subplts, 6))

    methods = ["original_image", "heat_map"]
    signs = ['all', 'positive' if np.all(attrs >= 0) else 'all']

    for idx in range(len(methods)):
        viz.visualize_image_attr(
            attrs,
            img,
            method=methods[idx],
            sign=signs[idx],
            cmap='viridis',
            show_colorbar=True,
            title=titles[idx] if titles is not None else None,
            plt_fig_axis=(fig, ax[idx]),
            use_pyplot=False,
            **kwargs
        )

    if auc_results:
        auc, pixels, scores = auc_results
        graph_auc_curve(pixels, scores, auc, title=titles[2], ax=ax[2])

    plt.show()

    return fig, ax
