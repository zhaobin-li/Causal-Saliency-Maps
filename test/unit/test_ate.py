from functools import partial

import torch

from causal.ate import ATE
from causal.estimators import get_reg_model, neyman_estimator
from causal.models import get_cnn_with_softmax
from causal.perturbers import bernoulli_trials_gen_batched, \
    balanced_trials_gen_batched, permutation_gen_batched
from causal.similarities import equal_weighter, cosine_similarity_batched
from causal.superpixels import get_block_seg
from causal.utils import get_val_loader

batch_size = 50

cnn_model = get_cnn_with_softmax()

dl = get_val_loader(shuffle=True)

img, label = next(iter(dl))

y_num, x_num = 3, 3

seg = get_block_seg(img, y_num=y_num, x_num=x_num)


def test_change_interpretable_model():
    neyman_coefs = ATE(
        cnn_model,
        interpretable_model=neyman_estimator,
        similarity_func=cosine_similarity_batched,
        perturb_func=partial(bernoulli_trials_gen_batched, seed=0)
    ).attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num),
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate = ATE(
        cnn_model,
        interpretable_model=get_reg_model(),
        similarity_func=cosine_similarity_batched,
        perturb_func=partial(bernoulli_trials_gen_batched, seed=0)
    )

    reg_coefs = ate.attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num),
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate.interpretable_model_ = neyman_estimator

    neyman_coefs_set = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    assert torch.allclose(neyman_coefs, neyman_coefs_set)

    ate.interpretable_model_ = get_reg_model()

    reg_coefs_set = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    assert torch.allclose(reg_coefs, reg_coefs_set)


def test_change_similarity_func():
    cosine_coefs = ATE(
        cnn_model,
        interpretable_model=get_reg_model(),
        similarity_func=cosine_similarity_batched,
        perturb_func=partial(bernoulli_trials_gen_batched, seed=0)
    ).attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num),
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate = ATE(
        cnn_model,
        interpretable_model=get_reg_model(),
        similarity_func=equal_weighter,
        similarity_func_multiple=(equal_weighter, cosine_similarity_batched),
        perturb_func=partial(bernoulli_trials_gen_batched, seed=0)
    )

    uniform_coefs = ate.attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num),
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate.similarity_func_ = cosine_similarity_batched

    cosine_coefs_set = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    assert torch.allclose(cosine_coefs, cosine_coefs_set)

    ate.similarity_func_ = equal_weighter

    uniform_coefs_set = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    assert torch.allclose(uniform_coefs, uniform_coefs_set)


def test_change_perturb_func():
    bernoulli_coefs = ATE(
        cnn_model,
        interpretable_model=get_reg_model(),
        similarity_func=equal_weighter,
        perturb_func=partial(bernoulli_trials_gen_batched, seed=0)
    ).attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num) // 4,
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate = ATE(
        cnn_model,
        interpretable_model=get_reg_model(),
        similarity_func=equal_weighter,
        perturb_func=partial(balanced_trials_gen_batched, seed=0)
    )

    balanced_coefs = ate.attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num) // 2,
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    ate.perturb_func_ = (partial(bernoulli_trials_gen_batched, seed=0), 2 ** (x_num * y_num) // 4)

    bernoulli_coefs_set = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    assert torch.allclose(bernoulli_coefs, bernoulli_coefs_set)

    ate.perturb_func_ = (partial(balanced_trials_gen_batched, seed=0), 2 ** (x_num * y_num) // 2)

    balanced_coefs_set = ate.attribute(
        use_previous_results=True,
        return_input_shape=False)

    assert torch.allclose(balanced_coefs, balanced_coefs_set)


def test_is_asymptotic_permutation():
    ate = ATE(
        cnn_model,
        interpretable_model=get_reg_model(),
        similarity_func=equal_weighter,
        perturb_func=permutation_gen_batched
    )

    _ = ate.attribute(
        img,
        target=label,
        feature_mask=seg,
        n_samples=2 ** (x_num * y_num),
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=True
    )

    assert ate.is_asymptotic

# def test_is_asymptotic_bernoulli():
#     ate = ATE(
#         cnn_model,
#         interpretable_model=get_reg_model(),
#         similarity_func=equal_weighter,
#         perturb_func=bernoulli_trials_gen_batched
#     )
#
#     for i in range(10):
#         _ = ate.attribute(
#             img,
#             target=label,
#             feature_mask=seg,
#             n_samples=2 ** (x_num * y_num) - 2,
#             perturbations_per_eval=batch_size,
#             return_input_shape=False,
#             show_progress=True
#         )
#
#     assert ate.is_asymptotic
