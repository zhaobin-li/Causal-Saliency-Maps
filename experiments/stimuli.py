import os
import shutil

import numpy as np
import torch
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import Occlusion
from captum.attr import Saliency
from captum.attr._utils import visualization as viz
from matplotlib import pyplot as plt
from tqdm import tqdm

from causal.auc import AUC
from causal.estimators import get_reg_model
from causal.explainers import eucl_lime_explainer
from causal.models import get_cnn_with_softmax
from causal.superpixels import get_superpixels
from causal.utils import get_top_1, get_val_loader, img_to_numpy, unnormalize, get_idx_to_label

# Prediction

# Deployment
num_imgs = 10
n_segments = 100
n_samples = 5000
device = torch.device("cuda:0")

torch.backends.cudnn.benchmark = True

plt_show = False

# num_imgs = 1
# n_segments = 10
# n_samples = 100
# device = torch.device("cpu")

# plt_show = True

# Intervention
auc_samples = 4
mode = 'deletion'
orders = ('highest', 'lowest')

batch_size = 500



img_path = "/projects/f_ps848_1/zhaobin/causal-saliency/experiment/img"
shutil.rmtree(img_path, ignore_errors=True)
os.makedirs(img_path, exist_ok=True)

cnn_model = get_cnn_with_softmax()
cnn_model.to(device)

reg_model = get_reg_model()

dl = get_val_loader(shuffle=True)

saliency_methods = dict(gradient=dict(func=Saliency(cnn_model),
                                      kwargs={}),
                        gradcam=dict(func=LayerGradCam(cnn_model, cnn_model[0].layer4[1].conv2),
                                     kwargs={}),
                        lime=dict(func=eucl_lime_explainer(cnn_model, interpretable_model=reg_model),
                                  kwargs=dict(n_samples=n_samples,
                                              perturbations_per_eval=batch_size,
                                              return_input_shape=True,
                                              show_progress=True)),
                        occlusion=dict(func=Occlusion(cnn_model),
                                       kwargs=dict(strides=(3, 15, 15),
                                                   sliding_window_shapes=(3, 30, 30))))

auc_cls = AUC(cnn_model)

for n_img, (img, label) in tqdm(enumerate(dl, 1), total=num_imgs):
    if n_img > num_imgs:
        break

    img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

    prob, label_idx = get_top_1(cnn_model, img)


    def attribute_image_features(algorithm, **kwargs):
        tensor_attributions = algorithm.attribute(img,
                                                  target=label_idx,
                                                  **kwargs)
        return tensor_attributions


    saliency_methods['lime']['kwargs']['feature_mask'] = get_superpixels(img, n_segments=n_segments, compactness=100,
                                                                         sigma=1,
                                                                         start_label=0)

    for method in saliency_methods:
        print(method)
        saliency_methods[method]['attr'] = attribute_image_features(saliency_methods[method]['func'],
                                                                    **saliency_methods[method]['kwargs'])
        if method == 'gradcam':
            saliency_methods[method]['attr'] = LayerAttribution.interpolate(saliency_methods[method]['attr'],
                                                                            img.shape[-2:])

    img_unnormed = img_to_numpy(unnormalize(img))

    fig, ax = plt.subplots(1, len(saliency_methods) + 1, figsize=(6 * len(saliency_methods) + 1, 6))

    gt = f"Label_{get_idx_to_label()[label.item()]}"
    ax[0].imshow(img_unnormed)
    ax[0].set_title(gt)
    ax[0].axis('off')

    pd = f"{get_idx_to_label()[label_idx.item()]}"
    for idx, method in enumerate(saliency_methods, 1):
        print(idx, method)

        attr = img_to_numpy(saliency_methods[method]['attr'])
        viz.visualize_image_attr(
            attr,
            original_image=None,
            method="heat_map",
            sign='positive' if np.all(attr >= 0) else 'all',
            cmap='viridis',
            show_colorbar=True,
            title=f"{method}_{pd}",
            plt_fig_axis=(fig, ax[idx]),
            use_pyplot=False
        )

    title = f"{gt}_Predicted_{pd}"
    fig.suptitle(title, fontsize=16)

    fig.savefig(os.path.join(img_path, f"Prediction_{title}.png"))

    if plt_show:
        plt.show()

    for method in saliency_methods:
        for order in orders:
            print(method)
            print(order)
            fig, ax = plt.subplots(1, auc_samples, figsize=(6 * auc_samples, 6))

            attrs = saliency_methods[method]['attr']

            if attrs.size(1) == 3:
                attrs = torch.mean(attrs, 1, keepdim=True)
            assert attrs.shape == (1, 1, 224, 224)

            if order == 'lowest':
                attrs = -1 * attrs

            *_, combined_model_inps = auc_cls.attribute(
                unnormalize(img),
                attrs=attrs,
                mode=mode,
                target=label_idx,
                n_samples=auc_samples,
                perturbations_per_eval=batch_size,
                show_progress=True,
                return_intermediate=True
            )

            for idx, partial_img in enumerate(combined_model_inps):
                ax[idx].imshow(img_to_numpy(partial_img))
                ax[idx].set_title(f"{1 - 1 / (auc_samples - 1) * idx:.2f}")
                ax[idx].axis('off')

            pd = f"{get_idx_to_label()[label_idx.item()]}"
            title = f"{method}_{order}_{mode}_{pd}"

            fig.suptitle(title, fontsize=16)
            fig.savefig(os.path.join(img_path, f"Intervention_{title}.png"))

            if plt_show:
                plt.show()
