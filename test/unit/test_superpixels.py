import torch
from captum.attr._core.lime import default_from_interp_rep_transform

from causal.superpixels import get_block_seg, batched_from_interp_rep_transform


def test_get_block_seg():
    img = torch.rand(1, 2, 3, 4)  # B*C*H*W
    seg = get_block_seg(img, 3, 2)
    truth = torch.tensor([[1, 1, 2, 2],
                          [3, 3, 4, 4],
                          [5, 5, 6, 6]]) - 1  # lime recommendation

    assert torch.equal(seg, truth)


def test_batched_from_interp_rep_transform():
    img = torch.rand(1, 2, 3, 4)  # B*C*H*W

    y_num = 2
    x_num = 3

    seg = get_block_seg(img, y_num, x_num)

    batch = 5
    samples = torch.randint(2, (batch, y_num * x_num))

    kwargs = {"feature_mask": seg, "baselines": torch.tensor(0)}

    model_inputs = []
    for sample in samples:
        model_inputs.append(default_from_interp_rep_transform(  # type: ignore
            sample.unsqueeze(0), img, **kwargs)
        )

    model_inputs = torch.cat(model_inputs)

    batched_model_inputs = batched_from_interp_rep_transform(samples, img, seg)

    assert batched_model_inputs.shape == (batch, *img[0].shape)
    assert torch.allclose(model_inputs, batched_model_inputs)