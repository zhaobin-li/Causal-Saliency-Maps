import torch
from captum.attr._core.lime import get_exp_kernel_similarity_function

from causal.similarities import cosine_similarity_batched, euclidean_similarity_batched


def test_cosine_similarity_batched():
    img = torch.rand(1, 2, 3, 4)

    batch = 5
    pert_imgs = torch.rand(batch, 2, 3, 4)

    cos_sim = get_exp_kernel_similarity_function()

    sims = []
    for pert_img in pert_imgs:
        sims.append(cos_sim(img, pert_img, None))
    sims = torch.tensor(sims)

    batched_sims = cosine_similarity_batched(img, pert_imgs)

    assert batched_sims.shape == torch.Size([batch])
    assert torch.allclose(sims, batched_sims)


def test_euclidean_distance_batched():
    img = torch.rand(1, 2, 3, 4)

    batch = 5
    pert_imgs = torch.rand(batch, 2, 3, 4)

    eucl_sim = get_exp_kernel_similarity_function('euclidean', kernel_width=1e3)

    sims = []
    for pert_img in pert_imgs:
        sims.append(eucl_sim(img, pert_img, None))
    sims = torch.tensor(sims)

    batched_sims = euclidean_similarity_batched(img, pert_imgs)

    assert batched_sims.shape == torch.Size([batch])
    assert torch.allclose(sims, batched_sims)
