from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from captum._utils.models import SkLearnLinearRegression
from captum.attr._utils import visualization as viz

from causal.ate import ATEBase, ATE
from causal.estimators import rise_estimator, neyman_estimator
from causal.models import get_cnn_with_softmax
from causal.perturbers import bernoulli_trials_gen_batched
from causal.perturbers import elementwise_product, permutation_gen_batched, balanced_trials_gen_batched
from causal.similarities import equal_weighter
from causal.superpixels import get_block_seg
from causal.utils import get_top_1, get_val_loader, get_idx_to_label, min_max_scale
from causal.utils import img_to_numpy, unnormalize
from tutorial.rise.explanations import RISE as OriginalRISE
from tutorial.rise.utils import tensor_imshow, get_class_name

batch_size = 500

cnn_model = get_cnn_with_softmax()

dl = get_val_loader(shuffle=True)

img, label = next(iter(dl))

img_unnormed = img_to_numpy(unnormalize(img))


def test_neyman_est_no_treatment_err():
    n_samples = 3

    num_interp_features = 2

    combined_outputs = torch.tensor([.1, .2, .3])

    combined_sim = torch.tensor([1, 1, 1])

    with pytest.raises(AssertionError):
        combined_interp_inps = torch.tensor([[0, 0],
                                             [0, 0],
                                             [1, 0]])

        coefs = neyman_estimator(combined_interp_inps, combined_outputs, combined_sim, n_samples,
                                 num_interp_features)


def test_rise_est_no_treatment_err():
    n_samples = 3

    num_interp_features = 2

    combined_outputs = torch.tensor([.1, .2, .3])

    combined_sim = torch.tensor([1, 1, 1])

    with pytest.raises(AssertionError):
        combined_interp_inps = torch.tensor([[0, 0],
                                             [0, 0],
                                             [1, 0]])

        coefs = rise_estimator(combined_interp_inps, combined_outputs, combined_sim, n_samples,
                               num_interp_features)


def test_neyman_estimator():
    n_samples = 3

    num_interp_features = 2

    combined_outputs = torch.tensor([.1, .2, .3])

    combined_sim = torch.tensor([1, 1, 1])

    combined_interp_inps = torch.tensor([[0, 0],
                                         [1, 1],
                                         [1, 0]])

    truth = torch.tensor([.15, 0])

    coefs = neyman_estimator(combined_interp_inps, combined_outputs, combined_sim, n_samples,
                             num_interp_features)

    assert torch.allclose(coefs, truth)


def test_rise_scaled_not_eq_neyman_bernoulli():
    y_num, x_num = 2, 2

    seg = get_block_seg(img, y_num=y_num, x_num=x_num)

    ate = ATE(
        cnn_model,
        interpretable_model=rise_estimator,
        similarity_func=equal_weighter,
        perturb_func=bernoulli_trials_gen_batched
    )

    rise_coefs = ate.attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num) // 2,
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate.interpretable_model_ = neyman_estimator

    neyman_coefs = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    assert not torch.allclose(min_max_scale(rise_coefs), min_max_scale(neyman_coefs))


def test_rise_scaled_eq_neyman_balanced():
    y_num, x_num = 2, 2

    seg = get_block_seg(img, y_num=y_num, x_num=x_num)

    ate = ATE(
        cnn_model,
        interpretable_model=rise_estimator,
        similarity_func=equal_weighter,
        perturb_func=balanced_trials_gen_batched
    )

    rise_coefs = ate.attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num) // 2,
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate.interpretable_model_ = neyman_estimator

    neyman_coefs = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    assert torch.allclose(min_max_scale(rise_coefs), min_max_scale(neyman_coefs))
    assert torch.allclose(rise_coefs.max() - rise_coefs.min(), 0.5 * (neyman_coefs.max() - neyman_coefs.min()))


def test_neyman_not_eq_ols():
    y_num, x_num = 2, 2

    seg = get_block_seg(img, y_num=y_num, x_num=x_num)

    ate = ATE(
        cnn_model,
        interpretable_model=neyman_estimator,
        similarity_func=equal_weighter,
        perturb_func=balanced_trials_gen_batched
    )

    neyman_coefs = ate.attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num) // 2,
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate.interpretable_model_ = SkLearnLinearRegression()

    ols_coefs = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    with pytest.raises(AssertionError):
        assert torch.allclose(neyman_coefs, ols_coefs)


# def test_ate_additive():
#     y_num, x_num = 2, 2
#
#     seg = get_block_seg(img, y_num=y_num, x_num=x_num)
#
#     seg[seg == 4] = 3
#     print(seg)
#
#     ate = ATE(
#         cnn_model,
#         interpretable_model=neyman_estimator,
#         similarity_func=equal_weighter,
#         perturb_func=permutation_gen_batched
#     )
#
#     neyman_coefs = ate.attribute(
#         img,
#         target=label,
#         feature_mask=seg,
#         # n_samples=2 ** (x_num * y_num),
#         n_samples=2 ** 3,
#         perturbations_per_eval=batch_size,
#         return_input_shape=False,
#         show_progress=True
#     )
#
#     y_num, x_num = 2, 1
#
#     seg = get_block_seg(img, y_num=y_num, x_num=x_num)
#
#     ate2 = ATE(
#         cnn_model,
#         interpretable_model=neyman_estimator,
#         similarity_func=equal_weighter,
#         perturb_func=permutation_gen_batched
#     )
#
#     neyman_coefs2 = ate2.attribute(
#         img,
#         target=label,
#         feature_mask=seg,
#         n_samples=2 ** (x_num * y_num),
#         perturbations_per_eval=batch_size,
#         return_input_shape=False,
#         show_progress=True
#     )
#
#     print(neyman_coefs, neyman_coefs2)
#     print(neyman_coefs.sum(), neyman_coefs2.sum())


def test_neyman_eq_ols_asymptotic():
    y_num, x_num = 2, 2

    seg = get_block_seg(img, y_num=y_num, x_num=x_num)

    ate = ATE(
        cnn_model,
        interpretable_model=neyman_estimator,
        similarity_func=equal_weighter,
        perturb_func=permutation_gen_batched
    )

    neyman_coefs = ate.attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num),
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate.interpretable_model_ = SkLearnLinearRegression()

    ols_coefs = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    assert torch.allclose(neyman_coefs, ols_coefs)


@torch.no_grad()
def test_rise_eq_orig():
    """Code in Saliency.ipynb by RISE authors at https://github.com/eclique/RISE"""

    N = 30
    p1 = 0.5
    s = 2

    batch_size = 10
    input_size = (224, 224)

    # Original RISE
    explainer = OriginalRISE(cnn_model, input_size=input_size, gpu_batch=batch_size)

    maskspath = 'masks.npy'
    explainer.generate_masks(N=N, s=s, p1=p1, savepath=maskspath)

    explainer.masks = explainer.masks.cpu()

    top_k = 1

    saliency = explainer(img).cpu().numpy()
    p, c = torch.topk(cnn_model(img), k=top_k)

    p, c = p[0], c[0]

    plt.figure(figsize=(10, 5 * top_k))
    for k in range(top_k):
        plt.subplot(top_k, 2, 2 * k + 1)
        plt.axis('off')
        plt.title('{:.2f}% {}'.format(100 * p[k], get_class_name(c[k])))
        tensor_imshow(img[0])

        plt.subplot(top_k, 2, 2 * k + 2)
        plt.axis('off')
        plt.title(get_class_name(c[k]))
        sal = saliency[c[k]]
        tensor_imshow(img[0])
        plt.imshow(sal, cmap='viridis', alpha=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.show()

    # Captum RISE
    explainer_cls = ATEBase(cnn_model,
                            interpretable_model=partial(rise_estimator, treatment_probs=p1),
                            similarity_func=equal_weighter,
                            similarity_func_multiple=None,
                            perturb_func=lambda *args, **kwargs: explainer.masks.reshape(N, -1),
                            perturb_interpretable_space=True,
                            from_interp_rep_transform=partial(elementwise_product, batched_sample_channels=1),
                            to_interp_rep_transform=None)

    prob, label_idx = get_top_1(cnn_model, img)

    assert torch.allclose(p[0], prob)
    assert torch.allclose(c[0], label_idx)

    attrs = explainer_cls.attribute(
        img,
        target=label_idx,
        n_samples=N,
        perturbations_per_eval=batch_size,
        show_progress=True,
        num_interp_features=224 * 224,
    ).squeeze(0)

    attrs = attrs.cpu().numpy().reshape(input_size)
    assert np.allclose(attrs, saliency[c])

    save_name = "CapRISE"
    gt = f"Label_{label.item()}-{get_idx_to_label()[label.item()]}"
    pd = f"{save_name}_{label_idx.item()}-{get_idx_to_label()[label_idx.item()]}"

    fig, ax = viz.visualize_image_attr_multiple(
        np.expand_dims(attrs, -1),  # adjust shape to height, width, channels
        img_unnormed,
        methods=["original_image", "heat_map"],
        signs=['all', 'positive'],
        cmap='viridis',
        show_colorbar=True,
        titles=[gt, pd]
    )
