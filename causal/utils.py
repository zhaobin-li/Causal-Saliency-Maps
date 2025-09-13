import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as transforms, datasets as datasets


def get_top_1(cnn_model, img_tensor):
    probs = cnn_model(img_tensor).squeeze(0)
    val, idx = torch.max(probs, 0, keepdim=True)
    return val, idx


def get_val_loader(valdir='/projects/f_ps848_1/imagenet/ilsvrc2012/validation_images', shuffle=False,
                   return_path=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tsfm = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

    dataset = ImageFolderWithPaths(valdir, tsfm) if return_path else datasets.ImageFolder(valdir, tsfm)

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=shuffle,
        num_workers=0, pin_memory=True)

    return val_loader


def get_idx_to_label():
    # to store the dictionary once loaded
    if not hasattr(get_idx_to_label, 'idx_to_labels'):
        labels_path = '/projects/f_ps848_1/imagenet/ilsvrc2012/imagenet_class_index.json'
        with open(labels_path) as json_data:
            get_idx_to_label.idx_to_labels = {int(idx): label for idx, [_, label] in json.load(json_data).items()}
    return get_idx_to_label.idx_to_labels


def print_result(probs, topk=1):
    idx_to_labels = get_idx_to_label()

    probs, label_indices = torch.topk(probs, topk)
    probs = probs.tolist()
    label_indices = label_indices.tolist()
    for prob, idx in zip(probs, label_indices):
        label = idx_to_labels[idx]
        print(f'{label} ({idx}):', round(prob, 4))


def unnormalize(img_, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Unnormalize ImageNet image according to Joel Simon at
    https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3"""
    img = img_.detach().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def img_to_numpy(img_tensor):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    return img_tensor.detach().permute(1, 2, 0).cpu().numpy()


def min_max_scale(input_tensor):
    max_value = torch.max(input_tensor)
    min_value = torch.min(input_tensor)

    if max_value == min_value:  # prevent divide by zero
        input_tensor = input_tensor / max_value  # equals torch.ones_like(input_tensor)
    else:  # min-max scaling
        input_tensor = (input_tensor - min_value) / (max_value - min_value)
    return input_tensor


def get_size(arr_tensor):
    return arr_tensor.element_size() * arr_tensor.nelement() / 1e9  # in GB


nested_dict = lambda: defaultdict(nested_dict)  # recursive defaultdict


def inverse_to_unique_idx(inverse_indices):
    unique_indices = torch.empty_like(inverse_indices)[:(inverse_indices.max() + 1)]
    for orig_idx, uniq_idx in enumerate(inverse_indices):
        unique_indices[uniq_idx] = orig_idx
    return unique_indices


def get_index_batched(arr, ele):
    """
    return ele_idx, element indices in arr, and corresponding arr_idx
    """
    # ele.shape == (n_samples, num_interp_features)
    # arr.shape == (n_stored_samples, num_interp_features)

    # n_samples = ele.size(0)
    # n_stored_samples = arr.size(0)
    #
    # ele = ele.t().unsqueeze(0).repeat(n_stored_samples, 1, 1)
    # arr = arr.unsqueeze(-1).repeat(1, 1, n_samples)

    ele = ele.t().unsqueeze(0)
    arr = arr.unsqueeze(-1)

    arr_idx, ele_idx = torch.nonzero(torch.all(torch.eq(arr, ele), dim=1), as_tuple=True)

    if len(ele_idx) > 0:
        ele_idx, inverse_indices = torch.unique(ele_idx, sorted=True, return_inverse=True)
        unique_indices = inverse_to_unique_idx(inverse_indices)
        arr_idx = arr_idx[unique_indices]

        return arr_idx, ele_idx

    else:
        return None, None


def non_index(arr, not_idx):
    mask = torch.ones(len(arr), device=arr.device, dtype=bool)
    mask[not_idx] = False
    return arr[mask]


class ImageFolderWithPaths(datasets.ImageFolder):
    # Ref: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d,
    def __getitem__(self, index):
        sample, target = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.samples[index][0]

        return sample, target, path


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
