"""Implement RISE with superpixel option using captum LIME"""
import torch
from captum._utils.models import SkLearnLinearRegression


# input:
# combined_interp_inps (X), combined_outputs (Y), combined_sim (weights) and
# kwargs including n_samples, baselines, feature_mask, num_interp_features
# return:
# saliency tensor


def neyman_estimator(combined_interp_inps, combined_outputs, combined_sim, n_samples,
                     num_interp_features=None, **kwargs):
    for i in range(num_interp_features):  # both treatment and control are present
        assert not (torch.all(combined_interp_inps[:, i] == 0) or
                    torch.all(combined_interp_inps[:, i] == 1))

    weighted_outputs = combined_sim * combined_outputs

    return torch.tensor(
        [weighted_outputs[combined_interp_inps[:, i] == 1].mean() - weighted_outputs[
            combined_interp_inps[:, i] == 0].mean()
         for i in range(num_interp_features)]).to(combined_outputs.device).reshape(1, num_interp_features)


def rise_estimator(combined_interp_inps, combined_outputs, combined_sim, n_samples,
                   num_interp_features, treatment_probs=None, **kwargs):
    """:param set treatment_probs to scaler E[M] (default 0.5) like rise equation 6, and if None uses MC E[M] vector.
    combined_sim not in original rise equation"""

    for i in range(num_interp_features):  # treatment present
        assert torch.any(combined_interp_inps[:, i] == 1)

    if treatment_probs is None:
        treatment_probs = combined_interp_inps.float().mean(axis=0)
        assert treatment_probs.shape == torch.Size([num_interp_features])

    sal = ((combined_sim * combined_outputs) @ combined_interp_inps.float()) / (
            n_samples * treatment_probs * combined_sim.mean())

    if isinstance(treatment_probs, torch.Tensor):
        assert torch.all(sal <= 1) and torch.all(sal >= 0)
    else:
        assert torch.all(sal <= (1 / treatment_probs)) and torch.all(sal >= 0)

    return sal.to(combined_outputs.device).reshape(1, num_interp_features)


def get_reg_model():
    model = SkLearnLinearRegression()
    # model = SkLearnLasso(alpha=0.1)
    return model
