from functools import partial

from causal.ate import ATE
from causal.estimators import rise_estimator
from causal.perturbers import balanced_trials_gen_batched
from causal.perturbers import bernoulli_trials_gen_batched
from causal.similarities import equal_weighter, cosine_similarity_batched, euclidean_similarity_batched


def get_ate_explainer(cnn_model, interpretable_model=None, similarity_func=None, perturb_func=None):
    return ATE(
        cnn_model,
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
        perturb_func=perturb_func
    )


def bern_rise_explainer(cnn_model, interpretable_model, seed=None, **kwargs):
    return get_ate_explainer(cnn_model, interpretable_model, similarity_func=equal_weighter,
                             perturb_func=partial(bernoulli_trials_gen_batched, seed=seed))


def bal_rise_explainer(cnn_model, interpretable_model, seed=None, **kwargs):
    return get_ate_explainer(cnn_model, interpretable_model, similarity_func=equal_weighter,
                             perturb_func=partial(balanced_trials_gen_batched, seed=seed))


def cos_lime_explainer(cnn_model, interpretable_model, seed=None, **kwargs):
    return get_ate_explainer(cnn_model, interpretable_model,
                             similarity_func=cosine_similarity_batched,
                             perturb_func=partial(bernoulli_trials_gen_batched, seed=seed))


def eucl_lime_explainer(cnn_model, interpretable_model, seed=None, **kwargs):
    return get_ate_explainer(cnn_model, interpretable_model,
                             similarity_func=euclidean_similarity_batched,
                             perturb_func=partial(bernoulli_trials_gen_batched, seed=seed))


def orig_rise_explainer(cnn_model, seed=None, **kwargs):
    return get_ate_explainer(cnn_model, interpretable_model=rise_estimator, similarity_func=equal_weighter,
                             perturb_func=partial(bernoulli_trials_gen_batched, seed=seed))
