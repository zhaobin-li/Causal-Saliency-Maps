import torch

from causal.utils import img_to_numpy, unnormalize, get_index_batched, inverse_to_unique_idx


def test_unnorm_no_se():
    """no side effect"""
    img = torch.rand(1, 3, 5, 10)
    img_unnormed = img_to_numpy(unnormalize(img))

    assert not torch.equal(torch.from_numpy(img_unnormed).permute(2, 0, 1).unsqueeze(0), img)


def test_get_index_batched():
    ele = torch.tensor([[0, 0], [1, 1], [1, 0], [1, 1]])
    arr = torch.tensor([[0, 0], [0, 1], [0, 0], [1, 1]])

    ele_idx_ground_truth = torch.tensor([0, 1, 3])
    arr_idx_ground_truth = torch.tensor([2, 3, 3])

    arr_idx, ele_idx = get_index_batched(arr, ele)

    torch.allclose(ele_idx, ele_idx_ground_truth)
    torch.allclose(arr_idx, arr_idx_ground_truth)


def test_get_index_batched_empty():
    ele = torch.tensor([[1, 0]])
    arr = torch.tensor([[0, 0], [0, 1], [0, 0], [1, 1]])

    arr_idx, ele_idx = get_index_batched(arr, ele)

    assert arr_idx is None
    assert ele_idx is None


def test_inverse_to_unique_idx():
    original = torch.tensor([1, 3, 2, 3])
    unique, inverse_indices = torch.unique(
        original, return_inverse=True)
    unique_indices = inverse_to_unique_idx(inverse_indices)

    assert torch.allclose(original[unique_indices], unique)
