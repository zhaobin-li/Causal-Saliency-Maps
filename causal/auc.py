"""Based on LIME to get insertion and deletion AUC metric as proposed in https://github.com/eclique/RISE"""

import inspect
import math
import warnings
from typing import Any, Callable, Tuple, Union, cast

import numpy as np
import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_input,
    _is_tuple,
)
from captum._utils.progress import progress
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.lime import LimeBase, default_from_interp_rep_transform
from captum.attr._utils.common import (
    _construct_default_feature_mask,
    _format_input_baseline,
)
from captum.log import log_usage
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import auc as sklearn_auc
from torch import Tensor


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen // 2, klen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


class AUCBase(LimeBase):

    def __init__(
            self,
            forward_func: Callable,
    ) -> None:

        LimeBase.__init__(self, forward_func=forward_func, interpretable_model=None, similarity_func=None,
                          perturb_func=default_ins_del_func,
                          perturb_interpretable_space=True,
                          from_interp_rep_transform=default_from_interp_rep_transform,
                          to_interp_rep_transform=None)

    @log_usage()
    def attribute(
            self,
            inputs: TensorOrTupleOfTensorsGeneric,
            attrs: Tensor,
            mode: str,
            target: TargetType = None,
            additional_forward_args: Any = None,
            n_samples: int = 50,
            perturbations_per_eval: int = 1,
            show_progress: bool = False,
            return_intermediate=False,
            **kwargs,
    ) -> Union[tuple[Any, Tensor], tuple[Union[float, Any], Any, Any]]:
        with torch.no_grad():
            inp_tensor = (
                cast(Tensor, inputs) if isinstance(inputs, Tensor) else inputs[0]
            )
            device = inp_tensor.device

            # get baseline score
            curr_sample = torch.zeros(kwargs["num_interp_features"]).unsqueeze(0).to(inputs.device)
            curr_model_inputs = [self.from_interp_rep_transform(  # type: ignore
                curr_sample, inputs, **kwargs)]

            expanded_additional_args = _expand_additional_forward_args(
                additional_forward_args, len(curr_model_inputs)
            )
            expanded_target = _expand_target(target, len(curr_model_inputs))
            model_out = self._evaluate_batch(
                curr_model_inputs,
                expanded_target,
                expanded_additional_args,
                device,
            )

            if model_out.item() > 0.1:
                warnings.warn(
                    f"Image has large baseline {model_out.item():.2f} probability"
                )

            interpretable_inps = []
            outputs = []

            model_inputs = []

            curr_model_inputs = []
            expanded_additional_args = None
            expanded_target = None
            perturb_generator = None
            batch_count = 0

            if inspect.isgeneratorfunction(self.perturb_func):
                perturb_generator = self.perturb_func(inputs, attrs, n_samples, mode, **kwargs)

            if show_progress:
                attr_progress = progress(
                    total=math.ceil(n_samples / perturbations_per_eval),
                    desc=f"{self.get_name()} attribution",
                )
                attr_progress.update(0)

            for _ in range(n_samples):
                if perturb_generator:
                    try:
                        curr_sample = next(perturb_generator)
                    except StopIteration:
                        warnings.warn(
                            "Generator completed prior to given n_samples iterations!"
                        )
                        break
                else:
                    curr_sample = self.perturb_func(inputs, attrs, n_samples, mode, **kwargs)

                batch_count += 1

                if self.perturb_interpretable_space:
                    interpretable_inps.append(curr_sample)
                    model_inp = self.from_interp_rep_transform(curr_sample, inputs, **kwargs)

                    curr_model_inputs.append(
                        model_inp
                    )

                    model_inputs.append(model_inp)
                else:
                    raise RuntimeError("AUC accepts only perturb_interpretable_space=True")

                if len(curr_model_inputs) == perturbations_per_eval:
                    if expanded_additional_args is None:
                        expanded_additional_args = _expand_additional_forward_args(
                            additional_forward_args, len(curr_model_inputs)
                        )
                    if expanded_target is None:
                        expanded_target = _expand_target(target, len(curr_model_inputs))

                    model_out = self._evaluate_batch(
                        curr_model_inputs,
                        expanded_target,
                        expanded_additional_args,
                        device,
                    )

                    if show_progress:
                        attr_progress.update()

                    outputs.append(model_out)

                    curr_model_inputs = []

            if len(curr_model_inputs) > 0:
                expanded_additional_args = _expand_additional_forward_args(
                    additional_forward_args, len(curr_model_inputs)
                )
                expanded_target = _expand_target(target, len(curr_model_inputs))
                model_out = self._evaluate_batch(
                    curr_model_inputs,
                    expanded_target,
                    expanded_additional_args,
                    device,
                )
                if show_progress:
                    attr_progress.update()
                outputs.append(model_out)

            if show_progress:
                attr_progress.close()

            combined_interp_inps = torch.cat(interpretable_inps)
            combined_outputs = (
                torch.cat(outputs)
                if len(outputs[0].shape) > 0
                else torch.stack(outputs)
            )
            combined_model_inps = torch.cat(model_inputs)

            auc_results = self._get_normalized_auc(combined_outputs, combined_interp_inps, n_samples, **kwargs)

            if return_intermediate:
                return (*auc_results, combined_model_inps)
            else:
                return auc_results

    def _get_normalized_auc(self, combined_outputs, combined_interp_inps, n_samples, feature_sizes=None,
                            num_interp_features=None,
                            **kwargs):

        assert combined_outputs.shape == torch.Size([n_samples])

        if num_interp_features is not None:
            assert combined_interp_inps.shape == (n_samples, num_interp_features)

        if feature_sizes is not None:
            assert feature_sizes[0].shape == torch.Size([num_interp_features])
        else:
            feature_sizes = torch.ones_like(combined_interp_inps[0]).unsqueeze(0)

        num_pixels = (combined_interp_inps.float() @ feature_sizes[0].float())  # matmul
        total_pixels = torch.sum(feature_sizes[0]).cpu().numpy()

        norm_auc = sklearn_auc(num_pixels.cpu().numpy(), combined_outputs.cpu().numpy()) / total_pixels

        assert (norm_auc >= 0 and norm_auc <= 1)
        return norm_auc, num_pixels.cpu().numpy(), combined_outputs.cpu().numpy()


def default_ins_del_func(_original_input, attrs, n_samples, mode, **kwargs):
    """:param attrs is tensor"""
    assert mode in ['insertion', 'deletion']

    attrs = attrs.squeeze().cpu().detach().numpy().ravel()
    order = np.argsort(attrs)[::-1]  # descending

    for end_point in np.linspace(0, len(attrs), num=n_samples, dtype=np.int_):
        partial_sample = np.zeros_like(attrs, dtype=np.int_)  # deletion ones_like
        partial_sample[order[:end_point]] = 1  # deletion = 0

        if mode == 'deletion':
            yield torch.from_numpy(1 - partial_sample).unsqueeze(0).to(_original_input.device)
        else:
            yield torch.from_numpy(partial_sample).unsqueeze(0).to(_original_input.device)


def construct_feature_mask_w_sizes(feature_mask, formatted_inputs):
    if feature_mask is None:
        feature_mask, num_interp_features = _construct_default_feature_mask(
            formatted_inputs
        )

    else:
        feature_mask = _format_input(feature_mask)
        min_interp_features = int(
            min(torch.min(single_inp).item() for single_inp in feature_mask)
        )
        if min_interp_features != 0:
            warnings.warn(
                "Minimum element in feature mask is not 0, shifting indices to"
                " start at 0."
            )
            feature_mask = tuple(
                single_inp - min_interp_features for single_inp in feature_mask
            )

        num_interp_features = int(
            max(torch.max(single_inp).item() for single_inp in feature_mask) + 1
        )

    feature_sizes = tuple(torch.bincount(torch.flatten(single_inp)) for single_inp in feature_mask)

    return feature_mask, num_interp_features, feature_sizes


class AUC(AUCBase):

    def __init__(
            self,
            forward_func: Callable,
    ) -> None:
        AUCBase.__init__(
            self,
            forward_func=forward_func,
        )

    @log_usage()
    def attribute(  # type: ignore
            self,
            inputs: TensorOrTupleOfTensorsGeneric,
            attrs: Tensor,
            mode: str,
            baselines: BaselineType = None,
            target: TargetType = None,
            additional_forward_args: Any = None,
            feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
            n_samples: int = None,
            perturbations_per_eval: int = 1,
            show_progress: bool = False,
            return_intermediate=False,
            **kwargs,
    ) -> (float, np.ndarray, np.ndarray):
        if n_samples is None:
            n_samples = torch.numel(attrs) + 1

        if n_samples > 10000:
            raise RuntimeError("n_samples > 10000")

        is_inputs_tuple = _is_tuple(inputs)
        formatted_attrs, baselines = _format_input_baseline(attrs, baselines)

        feature_mask, num_interp_features, feature_sizes = construct_feature_mask_w_sizes(
            feature_mask, formatted_attrs
        )

        try:
            torch.broadcast_tensors(feature_mask[0], inputs)
        except RuntimeError:
            raise ValueError("feature mask not broadcastable to inputs")

        attrs: Tensor

        return super().attribute.__wrapped__(
            self,
            inputs=inputs,
            attrs=attrs,
            mode=mode,
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
            feature_sizes=feature_sizes,
            show_progress=show_progress,
            return_intermediate=return_intermediate,
            **kwargs,
        )
