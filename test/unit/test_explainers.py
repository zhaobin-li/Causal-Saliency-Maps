from functools import partial

import torch
from captum.attr._core.lime import get_exp_kernel_similarity_function, Lime

from causal.estimators import get_reg_model
from causal.explainers import cos_lime_explainer
from causal.models import get_cnn_with_softmax
from causal.perturbers import bernoulli_trials_gen
from causal.superpixels import get_block_seg
from causal.utils import get_val_loader

batch_size = 50

cnn_model = get_cnn_with_softmax()

reg_model = get_reg_model()

dl = get_val_loader(shuffle=True)

img, label = next(iter(dl))

y_num, x_num = 2, 3

seg = get_block_seg(img, y_num=y_num, x_num=x_num)


def test_lime_eq_orig_batched():
    coefs = cos_lime_explainer(cnn_model, interpretable_model=reg_model, seed=0).attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num),
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True,
    )

    coefs_original = Lime(
        cnn_model,
        interpretable_model=reg_model,
        similarity_func=get_exp_kernel_similarity_function(),
        perturb_func=partial(bernoulli_trials_gen, n_samples=2 ** (x_num * y_num), seed=0)).attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num),
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True,
    )

    torch.equal(coefs, coefs_original)
