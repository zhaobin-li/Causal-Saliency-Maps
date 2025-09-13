import torch

from causal.perturbers import balanced_trials_gen


def test_balanced_trials():
    n_samples = 100
    n_features = 50
    gen = balanced_trials_gen(torch.zeros(n_features), n_samples=n_samples,
                              num_interp_features=n_features)

    sample_lst = []
    for sample in gen:
        assert isinstance(sample, torch.LongTensor)
        assert sample.shape == (1, n_features)

        sample_lst.append(sample)

    sample_mat = torch.cat(sample_lst)
    assert sample_mat.shape == (n_samples, n_features)

    # check is balanced
    sum_col = torch.sum(sample_mat, axis=0)
    assert torch.all(sum_col == int(n_samples / 2))

    # check is randomized
    sum_row = torch.sum(sample_mat, axis=1)
    assert not (torch.all(sum_row[:int(n_samples / 2)] == 0) and torch.all(sum_row[:int(n_samples / 2)] == 1))
