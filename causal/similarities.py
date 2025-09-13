import torch
import torch.nn.functional as F


def equal_weighter(original_inp, perturbed_inp, **kwargs):
    return torch.ones(len(perturbed_inp)).to(original_inp.device)


def cosine_similarity_batched(original_inp, batched_perturbed_inp, kernel_width: float = 1.0, **kwargs):
    distance = 1 - F.cosine_similarity(original_inp.reshape(1, -1),
                                       batched_perturbed_inp.reshape(batched_perturbed_inp.size(0), -1), dim=1)

    return torch.exp(-1 * (distance ** 2) / (2 * (kernel_width ** 2))).to(original_inp.device)


def euclidean_similarity_batched(original_inp, batched_perturbed_inp, kernel_width: float = 1e3, **kwargs):
    distance = 1 - F.pairwise_distance(original_inp.reshape(1, -1),
                                       batched_perturbed_inp.reshape(batched_perturbed_inp.size(0), -1), p=2)

    return torch.exp(-1 * (distance ** 2) / (2 * (kernel_width ** 2))).to(original_inp.device)
