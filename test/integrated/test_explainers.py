import os

import matplotlib.pyplot as plt
import pytest
import torch
from skimage.segmentation import mark_boundaries

from causal.estimators import get_reg_model
from causal.explainers import bern_rise_explainer, bal_rise_explainer, cos_lime_explainer, eucl_lime_explainer, \
    orig_rise_explainer
from causal.graphs import graph_saliency_map
from causal.models import get_cnn_with_softmax
from causal.superpixels import get_superpixels
from causal.utils import get_val_loader, get_idx_to_label, img_to_numpy, unnormalize, get_top_1

# Deployment
# repeat = 2
# n_segments = 100
# n_samples = 5000
# device = torch.device("cuda:0")

repeat = 1
n_segments = 10
n_samples = 100
device = torch.device("cpu")

batch_size = 500

img_path = "/projects/f_ps848_1/zhaobin/causal-saliency/img"
os.makedirs(img_path, exist_ok=True)

cnn_model = get_cnn_with_softmax()
cnn_model.to(device)

reg_model = get_reg_model()

dl = get_val_loader(shuffle=True)

img, label = next(iter(dl))
img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

prob, label_idx = get_top_1(cnn_model, img)

seg = get_superpixels(img, n_segments=n_segments, compactness=100, sigma=1, start_label=0)

img_unnormed = img_to_numpy(unnormalize(img))


def test_viz_img():
    title_str = f"Label_{get_idx_to_label()[label.item()]}"

    plt.imshow(img_to_numpy(unnormalize(img)))
    plt.title(title_str)
    plt.axis('off')

    plt.savefig(os.path.join(img_path, f"{title_str}.png"))
    plt.show()


def test_viz_seg():
    title_str = f"Label_{get_idx_to_label()[label.item()]}_SLIC"

    plt.imshow(mark_boundaries(img_to_numpy(unnormalize(img)), seg.cpu().numpy()))
    plt.title(title_str)
    plt.axis('off')

    plt.savefig(os.path.join(img_path, f"{title_str}.png"))
    plt.show()


@pytest.mark.parametrize("explainer_f, save_name", [(bern_rise_explainer, "BernRise"),
                                                    (bal_rise_explainer, "BalRise"),
                                                    (cos_lime_explainer, "CosLime"),
                                                    (eucl_lime_explainer, "EuclLime"),
                                                    (orig_rise_explainer, "OrigRise")])
@pytest.mark.parametrize('count', range(repeat))
def test_explainers(explainer_f, save_name, count):
    explainer_cls = explainer_f(cnn_model, interpretable_model=reg_model)

    attrs = explainer_cls.attribute(
        img,
        target=label_idx,
        feature_mask=seg,
        n_samples=n_samples,
        perturbations_per_eval=batch_size,
        return_input_shape=True,
        show_progress=True,
    )

    gt = f"Label_{get_idx_to_label()[label.item()]}"
    pd = f"{save_name}_{get_idx_to_label()[label_idx.item()]}"

    for mode in ['insertion', 'deletion']:
        super_pixel_auc, pixels, scores = explainer_cls.get_auc(mode)

        fig, ax = graph_saliency_map(attrs, img_unnormed, titles=[gt, pd, f"{mode[0]}AUC"],
                                     auc_results=(super_pixel_auc, pixels, scores))

        res_str = f"{gt}_{pd}_{mode[0]}AUC"
        fig.savefig(os.path.join(img_path, f"{res_str}.png"))
