import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import conv2d

from causal.auc import AUC, default_ins_del_func
from causal.estimators import get_reg_model
from causal.explainers import bern_rise_explainer
from causal.graphs import graph_saliency_map
from causal.models import get_cnn_with_softmax
from causal.superpixels import get_block_seg
from causal.utils import get_top_1, get_val_loader, img_to_numpy, unnormalize, get_idx_to_label
from tutorial.rise.evaluation import CausalMetric as RISECausalMetric, auc as RISEauc, gkern

device = torch.device("cpu")

batch_size = 50

img_path = "/projects/f_ps848_1/zhaobin/causal-saliency/img"
os.makedirs(img_path, exist_ok=True)

cnn_model = get_cnn_with_softmax()
cnn_model.to(device)

reg_model = get_reg_model()

dl = get_val_loader(shuffle=True)

img, label = next(iter(dl))
img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

# y_num and x_num has to divide 224 evenly to make the comparison True
# since Captum AUC uses linspace but Rise AUC uses quotient division
y_num, x_num = 2, 2
seg, seg_size = get_block_seg(img, y_num=y_num, x_num=x_num, return_size=True)

img_unnormed = img_to_numpy(unnormalize(img))

n_samples = 100


def test_default_ins_func():
    mode = 'insertion'

    attrs = torch.tensor([1, 6, 3, 5])

    n_samples = 3

    expected = torch.tensor([[[0, 0, 0, 0]],
                             [[0, 1, 0, 1]],
                             [[1, 1, 1, 1]]]).to(device)

    _original_input = torch.rand(1, 3, 5, 10).to(device)

    for idx, coefs in enumerate(default_ins_del_func(img, attrs, n_samples, mode)):
        torch.equal(coefs, expected[idx])


def test_default_del_func():
    mode = 'deletion'

    attrs = torch.tensor([1, 6, 3, 5])

    n_samples = 3

    expected = torch.tensor([[[1, 1, 1, 1]],
                             [[1, 0, 1, 0]],
                             [[0, 0, 0, 0]]]).to(device)

    _original_input = torch.rand(1, 3, 5, 10).to(device)

    for idx, coefs in enumerate(default_ins_del_func(img, attrs, n_samples, mode)):
        torch.equal(coefs, expected[idx])


def test_orig_auc_equal():
    """Code in Evaluation.ipynb by RISE authors at https://github.com/eclique/RISE"""
    explainer_cls = bern_rise_explainer(cnn_model, interpretable_model=reg_model)

    prob, label_idx = get_top_1(cnn_model, img)

    coefs, attrs = explainer_cls.attribute(
        img,
        target=label_idx,
        feature_mask=seg,
        n_samples=n_samples,
        perturbations_per_eval=batch_size,
        show_progress=True,
        return_both=True
    )

    klen = 11
    ksig = 5

    kern = gkern(klen, ksig)  # (3, 3, 11, 11)
    blur = lambda x: nn.functional.conv2d(x, kern.to(x.device), padding=klen // 2)

    for mode in ['insertion', 'deletion']:
        baseline_fn = torch.zeros_like if mode == 'deletion' else blur

        # Original RISE

        # whole image auc
        cnn_model.cuda()  # in place transfer to GPU needed by RISE

        metric = RISECausalMetric(cnn_model, mode[:3], seg_size,
                                  substrate_fn=baseline_fn)

        rise_auc = RISEauc(  # RISE uses trapezoidal rule like sklearn
            metric.single_run(img.cpu(), attrs.cpu().numpy(), verbose=1))

        cnn_model.to(device)

        print(f"Original RISE AUC: {rise_auc}")

        auc_cls = AUC(cnn_model)

        whole_image_auc, pixels, scores = auc_cls.attribute(
            img,
            attrs=torch.mean(attrs, 1, keepdim=True),
            mode=mode,
            target=label_idx,
            show_progress=True,
            n_samples=x_num * y_num + 1,
            baselines=baseline_fn(img)
        )

        print(f"Captum Whole Image AUC: {whole_image_auc}")

        # superpixel auc
        auc_cls = AUC(cnn_model)

        super_pixel_auc, pixels, scores = auc_cls.attribute(
            img,
            attrs=coefs,
            mode=mode,
            target=label_idx,
            feature_mask=seg,
            show_progress=True,
            baselines=baseline_fn(img)
        )

        print(f"Captum Superpixel AUC: {super_pixel_auc}")

        assert np.allclose(whole_image_auc, super_pixel_auc)
        assert np.allclose(rise_auc, super_pixel_auc)

        save_name = "BalRise"
        gt = f"Label_{label.item()}-{get_idx_to_label()[label.item()]}"
        pd = f"{save_name}_{label_idx.item()}-{get_idx_to_label()[label_idx.item()]}"

        fig, ax = graph_saliency_map(attrs, img_unnormed, titles=[gt, pd, f"{mode[0]}AUC"],
                                     auc_results=(super_pixel_auc, pixels, scores))

        res_str = f"{gt}_{pd}_{mode[0]}AUC"
        fig.savefig(os.path.join(img_path, f"{res_str}.png"))


def test_auc_in_ate():
    explainer_cls = bern_rise_explainer(cnn_model, interpretable_model=reg_model)

    prob, label_idx = get_top_1(cnn_model, img)

    coefs, attrs = explainer_cls.attribute(
        img,
        target=label_idx,
        feature_mask=seg,
        n_samples=n_samples,
        perturbations_per_eval=batch_size,
        show_progress=True,
        return_both=True
    )

    auc_cls = AUC(cnn_model)

    for mode in ['insertion', 'deletion']:
        super_pixel_auc, pixels, scores = auc_cls.attribute(
            img,
            attrs=coefs,
            mode=mode,
            target=label_idx,
            feature_mask=seg,
            show_progress=True
        )

        super_pixel_auc_in_ate, pixels_in_ate, scores_in_ate = explainer_cls.get_auc(mode)

        assert np.allclose(super_pixel_auc, super_pixel_auc_in_ate)
        assert np.allclose(pixels, pixels_in_ate)
        assert np.allclose(scores, scores_in_ate)
