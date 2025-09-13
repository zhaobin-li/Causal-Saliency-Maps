"""Based on LIME and
1. pass in n_samples argument to perturb_func through kwargs in ATEBase.attribute
2. return both coefs and salience with return_both parameter in ATE.attribute
2. expand interpretable_model to take in causal/estimators.py estimators in ATEBase.__init__"""

import math
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Tuple, Union, Iterable

import numpy as np
import torch
from captum._utils.common import (
    _is_tuple,
)
from captum._utils.models.model import Model
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.lime import LimeBase, Lime, construct_feature_mask
from captum.attr._utils.common import (
    _format_input_baseline,
)
from captum.log import log_usage
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from causal.auc import AUC
from causal.superpixels import batched_from_interp_rep_transform
from causal.utils import get_index_batched, non_index


class ATEBase(LimeBase):

    def __init__(
            self,
            forward_func: Callable,
            interpretable_model: Callable or Model,
            similarity_func: Callable,
            similarity_func_multiple: Optional[Iterable[Callable]],
            perturb_func: Callable,
            perturb_interpretable_space: bool,
            from_interp_rep_transform: Optional[Callable],
            to_interp_rep_transform: Optional[Callable],

    ) -> None:
        # ATE.__mro__ dictates calling super() will call Lime since both ATEBase and Lime inherits LimeBase
        # has to use LimeBase.__init__(self, ...) explicitly

        LimeBase.__init__(self, forward_func=forward_func, interpretable_model=interpretable_model,
                          similarity_func=similarity_func,
                          perturb_func=perturb_func,
                          perturb_interpretable_space=perturb_interpretable_space,
                          from_interp_rep_transform=from_interp_rep_transform,
                          to_interp_rep_transform=to_interp_rep_transform)

        self.similarity_func_multiple = similarity_func_multiple if similarity_func_multiple is not None else (
            similarity_func,)

        self.combined_interp_inps = None
        self.combined_outputs = None
        self.combined_sim = None

        self.stored_interp_inps = None
        self.stored_outputs = None
        self.stored_sim = None

        self.is_asymptotic = False

        self.samples = []
        self.reused = []

    @property
    def accum_samples(self):
        return sum(self.samples)

    @property
    def reused_samples(self):
        return sum(self.reused)

    @property
    def prev_num_samples(self):
        return self.samples[-1]

    @property
    def prev_reused_samples(self):
        return self.reused[-1]

    @log_usage()
    def attribute(
            self,
            inputs: TensorOrTupleOfTensorsGeneric = None,
            target: TargetType = None,
            additional_forward_args: Any = None,
            n_samples: int = 50,
            perturbations_per_eval: int = 1,
            show_progress: bool = False,
            use_previous_results: bool = False,
            **kwargs,
    ) -> Tensor:

        if use_previous_results:
            assert self.combined_interp_inps is not None \
                   and self.combined_outputs is not None \
                   and self.combined_sim is not None \
                   and self.kwargs is not None

        else:

            self.inputs = inputs
            self.target = target
            self.additional_forward_args = additional_forward_args
            self.n_samples = n_samples
            self.perturbations_per_eval = perturbations_per_eval
            self.kwargs = kwargs

            self.compute_and_store_res(
                self.inputs,
                self.target,
                self.additional_forward_args,
                self.n_samples,
                self.perturbations_per_eval,
                show_progress=show_progress,
                **self.kwargs)

        if isinstance(self.interpretable_model, Model):
            dataset = TensorDataset(
                self.combined_interp_inps, self.combined_outputs, self.combined_sim[self.similarity_func]
            )
            self.interpretable_model.fit(DataLoader(dataset, batch_size=len(dataset)))
            coefs = self.interpretable_model.representation().to(self.combined_interp_inps.device)
        else:
            coefs = self.interpretable_model(self.combined_interp_inps, self.combined_outputs,
                                             self.combined_sim[self.similarity_func],
                                             self.n_samples, **self.kwargs)

        assert coefs.shape == (1, self.kwargs['num_interp_features'])
        return coefs.float()

    @torch.no_grad()
    def compute_and_store_res(self,
                              inputs: TensorOrTupleOfTensorsGeneric,
                              target: TargetType,
                              additional_forward_args: Any,
                              n_samples: int,
                              perturbations_per_eval: int,
                              reuse_stored_samples: bool = False,
                              show_progress: bool = False,
                              **kwargs):

        assert self.perturb_interpretable_space and additional_forward_args is None
        if "baselines" in kwargs:
            assert kwargs["baselines"] == 0

        self.combined_interp_inps = self.perturb_func(inputs, n_samples,
                                                      **kwargs)
        self.samples.append(n_samples)

        if reuse_stored_samples and self.stored_interp_inps is not None:
            stored_idx, combined_idx = get_index_batched(self.stored_interp_inps, self.combined_interp_inps)
            if combined_idx is not None:
                self.combined_interp_inps = non_index(self.combined_interp_inps, combined_idx)
                self.reused.append(len(stored_idx))

        if len(self.combined_interp_inps) > 0:
            similarities = defaultdict(list)
            outputs = []

            batch_count = math.ceil(len(self.combined_interp_inps) / perturbations_per_eval)
            itr = tqdm(range(batch_count), desc=f"{self.get_name()} attribution") if show_progress else range(
                batch_count)

            for idx in itr:
                batched_interpretable_inps = self.combined_interp_inps[
                                             idx * perturbations_per_eval: min((idx + 1) * perturbations_per_eval,
                                                                               len(self.combined_interp_inps))]

                batched_model_inputs = self.from_interp_rep_transform(
                    batched_interpretable_inps, inputs, **kwargs)

                batched_outputs = self.forward_func(batched_model_inputs)
                outputs.append(batched_outputs[:, target].squeeze())

                for f in self.similarity_func_multiple:
                    similarities[f].append(f(
                        inputs, batched_model_inputs, **kwargs
                    ))

            self.combined_outputs = torch.cat(outputs)
            self.combined_sim = {f: torch.cat(similarities[f]) for f in self.similarity_func_multiple}

        if reuse_stored_samples and self.stored_interp_inps is not None and stored_idx is not None:
            if len(self.combined_interp_inps) > 0:
                self.combined_interp_inps = torch.cat(
                    (self.combined_interp_inps, self.stored_interp_inps[stored_idx]))
                self.combined_outputs = torch.cat((self.combined_outputs, self.stored_outputs[stored_idx]))
                self.combined_sim = {f: torch.cat((self.combined_sim[f], self.stored_sim[f][stored_idx])) for f in
                                     self.similarity_func_multiple}

                self.store_res(self.combined_interp_inps[:-len(stored_idx)],
                               self.combined_outputs[:-len(stored_idx)],
                               {f: self.combined_sim[f][:-len(stored_idx)] for f in
                                self.similarity_func_multiple}, **kwargs)
            else:
                self.combined_interp_inps = self.stored_interp_inps[stored_idx]
                self.combined_outputs = self.stored_outputs[stored_idx]
                self.combined_sim = {f: self.stored_sim[f][stored_idx] for f in
                                     self.similarity_func_multiple}
        else:
            self.store_res(self.combined_interp_inps, self.combined_outputs, self.combined_sim, **kwargs)

        assert self.combined_interp_inps.shape == (n_samples, kwargs['num_interp_features'])

        assert self.combined_outputs.shape == torch.Size([n_samples])
        assert all((self.combined_sim[f].shape == torch.Size([n_samples])) for f in
                   self.similarity_func_multiple)

        if not torch.all(self.combined_outputs <= 1) and torch.all(
                self.combined_outputs >= 0):
            warnings.warn('please use probabilities rather than logits')

        # print(f"{self.combined_interp_inps=}, {self.combined_outputs=}, {self.combined_sim=}")
        # print(f"{self.similarity_func=}, {self.similarity_func_multiple=}, {self.combined_sim=}, {self.stored_sim=}")

    # def compute_and_store_res(self,
    #                            inputs: TensorOrTupleOfTensorsGeneric,
    #                            target: TargetType,
    #                            additional_forward_args: Any,
    #                            n_samples: int,
    #                            perturbations_per_eval: int,
    #                            reuse_stored_samples: bool = False,
    #                            show_progress: bool = False,
    #                            **kwargs):
    #     with torch.no_grad():
    #         if reuse_stored_samples:
    #             combined_matching_idxs = []
    #
    #         inp_tensor = (
    #             cast(Tensor, inputs) if isinstance(inputs, Tensor) else inputs[0]
    #         )
    #         device = inp_tensor.device
    #
    #         interpretable_inps = []
    #         similarities = []
    #         outputs = []
    #
    #         model_inputs = []
    #
    #         curr_model_inputs = []
    #         expanded_additional_args = None
    #         expanded_target = None
    #         perturb_generator = None
    #         if inspect.isgeneratorfunction(self.perturb_func):
    #             perturb_generator = self.perturb_func(inputs, n_samples, **kwargs)
    #
    #         if show_progress:
    #             attr_progress = progress(
    #                 total=math.ceil(n_samples / perturbations_per_eval),
    #                 desc=f"{self.get_name()} attribution",
    #             )
    #             attr_progress.update(0)
    #
    #         batch_count = 0
    #
    #         for _ in range(n_samples):
    #             if perturb_generator:
    #                 try:
    #                     curr_sample = next(perturb_generator)
    #                 except StopIteration:
    #                     warnings.warn(
    #                         "Generator completed prior to given n_samples iterations!"
    #                     )
    #                     break
    #             else:
    #                 curr_sample = self.perturb_func(inputs, n_samples, **kwargs)
    #
    #             if reuse_stored_samples:
    #                 if self.perturb_interpretable_space:
    #                     curr_sample_matches = self.stored_interp_inps
    #                 else:
    #                     curr_sample_matches = self.stored_model_inps
    #
    #                 matching_idx = torch.nonzero(torch.all(torch.eq(curr_sample, curr_sample_matches), dim=1))
    #
    #                 if torch.numel(matching_idx) > 0:
    #                     combined_matching_idxs.append(matching_idx[0].item())
    #                     continue
    #
    #             if self.perturb_interpretable_space:
    #                 interpretable_inps.append(curr_sample)
    #                 curr_model_inputs.append(
    #                     self.from_interp_rep_transform(  # type: ignore
    #                         curr_sample, inputs, **kwargs
    #                     )
    #                 )
    #             else:
    #                 curr_model_inputs.append(curr_sample)
    #                 interpretable_inps.append(
    #                     self.to_interp_rep_transform(  # type: ignore
    #                         curr_sample, inputs, **kwargs
    #                     )
    #                 )
    #             curr_sim = self.similarity_func(
    #                 inputs, curr_model_inputs[-1], interpretable_inps[-1], **kwargs
    #             )
    #             similarities.append(
    #                 curr_sim.flatten()
    #                 if isinstance(curr_sim, Tensor)
    #                 else torch.tensor([curr_sim], device=device)
    #             )
    #
    #             if len(curr_model_inputs) == perturbations_per_eval:
    #                 batch_count += 1
    #                 print(f"{batch_count=}")
    #
    #                 if expanded_additional_args is None:
    #                     expanded_additional_args = _expand_additional_forward_args(
    #                         additional_forward_args, len(curr_model_inputs)
    #                     )
    #                 if expanded_target is None:
    #                     expanded_target = _expand_target(target, len(curr_model_inputs))
    #
    #                 model_out = self._evaluate_batch(
    #                     curr_model_inputs,
    #                     expanded_target,
    #                     expanded_additional_args,
    #                     device,
    #                 )
    #
    #                 if show_progress:
    #                     attr_progress.update()
    #
    #                 outputs.append(model_out)
    #
    #                 model_inputs.extend(curr_model_inputs)
    #
    #                 curr_model_inputs = []
    #
    #         if len(curr_model_inputs) > 0:
    #
    #             expanded_additional_args = _expand_additional_forward_args(
    #                 additional_forward_args, len(curr_model_inputs)
    #             )
    #             expanded_target = _expand_target(target, len(curr_model_inputs))
    #             model_out = self._evaluate_batch(
    #                 curr_model_inputs,
    #                 expanded_target,
    #                 expanded_additional_args,
    #                 device,
    #             )
    #             if show_progress:
    #                 attr_progress.update()
    #             outputs.append(model_out)
    #
    #             model_inputs.extend(curr_model_inputs)
    #
    #         if show_progress:
    #             attr_progress.close()
    #
    #         if interpretable_inps:  # not empty
    #             self.combined_interp_inps = torch.cat(interpretable_inps)
    #             self.combined_model_inps = torch.cat(model_inputs)
    #             self.combined_outputs = (
    #                 torch.cat(outputs)
    #                 if len(outputs[0].shape) > 0
    #                 else torch.stack(outputs)
    #             )
    #             self.combined_sim = (
    #                 torch.cat(similarities)
    #                 if len(similarities[0].shape) > 0
    #                 else torch.stack(similarities)
    #             )
    #
    #             if reuse_stored_samples and combined_matching_idxs:
    #                 matching_interp_inps = self.stored_interp_inps[combined_matching_idxs]
    #                 matching_model_inps = self.stored_model_inps[combined_matching_idxs]
    #                 matching_outputs = self.stored_outputs[combined_matching_idxs]
    #                 matching_sim = self.stored_sim[combined_matching_idxs]
    #
    #                 self.store_res(self.combined_interp_inps, self.combined_model_inps, self.combined_outputs,
    #                                self.combined_sim, **kwargs)
    #
    #                 self.combined_interp_inps = torch.cat(
    #                     [self.combined_interp_inps, matching_interp_inps])
    #                 self.combined_model_inps = torch.cat(
    #                     [self.combined_model_inps, matching_model_inps])
    #                 self.combined_outputs = torch.cat([self.combined_outputs, matching_outputs])
    #                 self.combined_sim = torch.cat([self.combined_sim, matching_sim])
    #
    #                 self.reused.append(len(combined_matching_idxs))
    #
    #             self.store_res(self.combined_interp_inps, self.combined_model_inps, self.combined_outputs,
    #                            self.combined_sim, **kwargs)
    #
    #         else:
    #             self.combined_interp_inps = self.stored_interp_inps[combined_matching_idxs]
    #             self.combined_model_inps = self.stored_model_inps[combined_matching_idxs]
    #             self.combined_outputs = self.stored_outputs[combined_matching_idxs]
    #             self.combined_sim = self.stored_sim[combined_matching_idxs]
    #
    #             self.reused.append(len(combined_matching_idxs))
    #
    #         self.samples.append(n_samples)
    #
    #         assert self.combined_interp_inps.shape == (n_samples, kwargs['num_interp_features'])
    #         assert self.combined_model_inps.shape == (n_samples, *self.inputs.shape[1:])
    #         assert self.combined_outputs.shape == torch.Size([n_samples])
    #         assert self.combined_sim.shape == torch.Size([n_samples])
    #
    #         if not torch.all(self.combined_outputs <= 1) and torch.all(
    #                 self.combined_outputs >= 0):
    #             warnings.warn('please use probabilities rather than logits')

    def store_res(self, combined_interp_inps, combined_outputs, combined_sim, num_interp_features,
                  **kwargs):

        if not self.is_asymptotic:
            if self.stored_interp_inps is None and self.stored_outputs is None and self.stored_sim is None:
                if len(combined_interp_inps) == 2 ** num_interp_features:  # self.perturb_func == 'perm' only
                    self.is_asymptotic = True
                    self.stored_interp_inps = combined_interp_inps
                    self.stored_outputs = combined_outputs
                    self.stored_sim = combined_sim

            # self.stored_interp_inps, inverse_indices = torch.unique(
            #     torch.cat([combined_interp_inps, self.stored_interp_inps]), return_inverse=True,
            #     dim=0)
            #
            # unique_indices = inverse_to_unique_idx(inverse_indices)
            #
            # self.stored_outputs = torch.cat([combined_outputs, self.stored_outputs])[unique_indices]
            # self.stored_model_inps = torch.cat([combined_model_inps, self.stored_model_inps])[unique_indices]
            # self.stored_sim = {f: torch.cat([combined_sim[f], self.stored_sim[f]])[unique_indices] for f in
            #                    self.similarity_func_multiple}
            #
            # stored_interp_inps_length = len(self.stored_interp_inps)
            #
            # assert len(self.stored_outputs) == stored_interp_inps_length
            # assert len(self.stored_model_inps) == stored_interp_inps_length
            # assert all((len(self.stored_sim[f]) == stored_interp_inps_length) for f in self.similarity_func_multiple)
            #
            # assert stored_interp_inps_length <= 2 ** num_interp_features
            #
            # if stored_interp_inps_length == 2 ** num_interp_features:
            #     self.is_asymptotic = True
            #
            # print(f"{get_size(self.stored_interp_inps)=}_"
            #       f"{get_size(self.stored_outputs)=}_{get_size(self.stored_sim)=}")

    @property
    def interpretable_model_(self):
        return self.interpretable_model

    @interpretable_model_.setter
    def interpretable_model_(self, value):
        self.interpretable_model = value

    @property
    def perturb_func_(self):
        return self.perturb_func

    @perturb_func_.setter
    def perturb_func_(self, value: Tuple[Callable, int] or Callable):
        try:
            perturb_func, self.n_samples = value
            self.perturb_func = perturb_func
        except TypeError:
            self.perturb_func = value

        if self.is_asymptotic and self.perturb_func == 'bern':
            stored_idx = torch.randint(len(self.stored_interp_inps), (self.n_samples,))

            self.combined_interp_inps = self.stored_interp_inps[stored_idx]
            self.combined_outputs = self.stored_outputs[stored_idx]
            self.combined_sim = {f: self.stored_sim[f][stored_idx] for f in
                                 self.similarity_func_multiple}

            self.reused.append(len(stored_idx))
            self.samples.append(self.n_samples)

        # elif self.is_asymptotic and self.perturb_func == 'perm':
        #     self.combined_interp_inps = self.stored_interp_inps
        #     self.combined_outputs = self.stored_outputs
        #     self.combined_sim = self.stored_sim
        #
        #     self.reused.append(self.n_samples)
        #     self.samples.append(self.n_samples)

        else:
            self.compute_and_store_res(self.inputs,
                                       self.target,
                                       self.additional_forward_args,
                                       self.n_samples,
                                       self.perturbations_per_eval,
                                       reuse_stored_samples=True,
                                       **self.kwargs)

    @property
    def similarity_func_(self):
        return self.similarity_func

    @similarity_func_.setter
    def similarity_func_(self, value: Tuple[Callable, bool] or Callable):
        try:
            similarity_func, recompute_stored = value
            self.similarity_func = similarity_func
        except TypeError:
            self.similarity_func = value
            recompute_stored = False

        if not recompute_stored:
            warnings.warn(f"recompute stored similarities is {recompute_stored}.")

        if self.similarity_func not in self.similarity_func_multiple:
            raise ValueError(f"{self.similarity_func} not in {self.similarity_func_multiple}")

        # similarities = []
        # for idx in range(len(self.combined_interp_inps)):
        #     curr_sim = self.similarity_func(
        #         self.inputs, self.combined_model_inps[idx], self.combined_interp_inps[idx], **self.kwargs
        #     )
        #     similarities.append(
        #         curr_sim.flatten()
        #         if isinstance(curr_sim, Tensor)
        #         else torch.tensor([curr_sim], device=self.combined_model_inps.device)
        #     )
        # 
        # combined_sim = (
        #     torch.cat(similarities)
        #     if len(similarities[0].shape) > 0
        #     else torch.stack(similarities)
        # )
        # 
        # self.combined_sim = combined_sim
        # 
        # if recompute_stored and self.stored_sim is not None:
        #     similarities = []
        #     for idx in range(len(self.stored_interp_inps)):
        #         curr_sim = self.similarity_func(
        #             self.inputs, self.stored_model_inps[idx], self.stored_interp_inps[idx], **self.kwargs
        #         )
        #         similarities.append(
        #             curr_sim.flatten()
        #             if isinstance(curr_sim, Tensor)
        #             else torch.tensor([curr_sim], device=self.stored_model_inps.device)
        #         )
        # 
        #     stored_sim = (
        #         torch.cat(similarities)
        #         if len(similarities[0].shape) > 0
        #         else torch.stack(similarities)
        #     )
        # 
        #     self.stored_sim = stored_sim


class ATE(ATEBase, Lime):

    def __init__(
            self,
            forward_func: Callable,
            interpretable_model: Callable = None,
            similarity_func: Optional[Callable] = None,
            similarity_func_multiple: Optional[Iterable[Callable]] = None,
            perturb_func: Optional[Callable] = None
    ) -> None:

        ATEBase.__init__(
            self,
            forward_func=forward_func,
            interpretable_model=interpretable_model,
            similarity_func=similarity_func,
            similarity_func_multiple=similarity_func_multiple,
            perturb_func=perturb_func,
            perturb_interpretable_space=True,
            from_interp_rep_transform=batched_from_interp_rep_transform,
            to_interp_rep_transform=None
        )

        self.cache = None

    def get_auc(self, mode, n_samples=None, show_progress=False) -> (float, np.ndarray, np.ndarray):
        assert mode in ['insertion', 'deletion']

        super_pixel_auc, pixels, scores = AUC(self.forward_func).attribute(
            inputs=self.inputs,
            attrs=self.coefs,
            mode=mode,
            baselines=self.baselines,
            target=self.target,
            additional_forward_args=self.additional_forward_args,
            feature_mask=self.feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=self.perturbations_per_eval,
            show_progress=show_progress
        )

        return super_pixel_auc, pixels, scores

    @log_usage()
    def attribute(  # type: ignore
            self,
            inputs: TensorOrTupleOfTensorsGeneric = None,
            baselines: BaselineType = None,
            target: TargetType = None,
            additional_forward_args: Any = None,
            feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
            n_samples: int = 25,
            perturbations_per_eval: int = 1,
            use_previous_results: bool = False,
            return_input_shape: bool = True,
            return_both: bool = False,
            show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:

        if use_previous_results:
            is_inputs_tuple, formatted_inputs, baselines, feature_mask, num_interp_features = self.cache

            self.coefs = ATEBase.attribute.__wrapped__(self, use_previous_results=use_previous_results)
        else:
            self.baselines = baselines
            self.feature_mask = feature_mask

            is_inputs_tuple = _is_tuple(inputs)
            formatted_inputs, baselines = _format_input_baseline(inputs, baselines)

            feature_mask, num_interp_features = construct_feature_mask(
                feature_mask, formatted_inputs
            )

            if num_interp_features > 10000:
                warnings.warn(
                    "Attempting to construct interpretable model with > 10000 features."
                    "This can be very slow or lead to OOM issues. Please provide a feature"
                    "mask which groups input features to reduce the number of interpretable"
                    "features. "
                )

            self.coefs = ATEBase.attribute.__wrapped__(
                self,
                inputs=inputs,
                target=target,
                additional_forward_args=additional_forward_args,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
                baselines=baselines if is_inputs_tuple else baselines[0],
                feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
                num_interp_features=num_interp_features,
                use_previous_results=use_previous_results,
                show_progress=show_progress,
            )

            self.cache = is_inputs_tuple, formatted_inputs, baselines, feature_mask, num_interp_features

        self.attrs = Lime._convert_output_shape(
            self,
            formatted_inputs,
            feature_mask,
            self.coefs,
            num_interp_features,
            is_inputs_tuple,
        )
        if return_both:
            return self.coefs, self.attrs
        elif return_input_shape:
            return self.attrs
        else:
            return self.coefs
