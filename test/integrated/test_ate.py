# pytest -s /projects/f_ps848_1/zhaobin/causal-saliency/test/integrated/test_ate.py

import socket
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch
from coverage.annotate import os
from tqdm import tqdm

from causal.ate import ATE
from causal.estimators import get_reg_model
from causal.estimators import neyman_estimator
from causal.models import get_cnn_with_softmax
from causal.perturbers import bernoulli_trials_gen_batched, \
    balanced_trials_gen_batched, permutation_gen_batched
from causal.similarities import equal_weighter
from causal.superpixels import get_block_seg
from causal.superpixels import get_superpixels
from causal.utils import get_top_1, nested_dict
from causal.utils import get_val_loader

data_path = "/projects/f_ps848_1/zhaobin/causal-saliency/data"
os.makedirs(data_path, exist_ok=True)

# if os.path.exists(data_path):
#     warnings.warn(f"Deleting {data_path}")
#     shutil.rmtree(data_path)

asymptotic_sim = False

testing = False

shuffle = True

cuda = 0

if torch.cuda.is_available():
    device = torch.device(f"cuda:{cuda}")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

print(f"{data_path=}, {testing=}, {device=}")

# batch_size = 50
# min_sample = 10
#
# num_imgs = 1
# num_reps = 2
#
# y_num, x_num = 3, 3
# seg_threshold = 5
#
# samples = np.array((0.75, 0.25))

batch_size = 512
min_sample = 100

num_imgs = 100
num_reps = 50

y_num, x_num = 7, 7
seg_threshold = 3

samples = np.array((10000, 1000, 100))

print(f"{min_sample=}, {num_imgs=}, {num_reps=}, {x_num=}={y_num=}, {samples=}, {seg_threshold=}")

if asymptotic_sim:
    max_nele = 30 * 1e9 / 4
    assert samples.max() * 2 ** (x_num * y_num - seg_threshold) * (x_num * y_num + seg_threshold) < max_nele

cnn_model = get_cnn_with_softmax()
cnn_model.to(device)

dl = get_val_loader(shuffle=shuffle, return_path=True)

sp_funcs = {"slic": partial(get_superpixels, n_segments=y_num * x_num, return_num_segs=True,
                            compactness=100, sigma=1, start_label=0),
            "blk": partial(get_block_seg, y_num=y_num, x_num=x_num, return_num_segs=True)}
if testing:
    pert_funcs = {"bal": (partial(balanced_trials_gen_batched, seed=0)),
                  "bern": (partial(bernoulli_trials_gen_batched, seed=0))}
else:
    # pert_funcs = {"bal": balanced_trials_gen_batched, "bern": bernoulli_trials_gen_batched}
    pert_funcs = {"bern": bernoulli_trials_gen_batched}

# sim_funcs = {"cos": cosine_similarity_batched, "unif": equal_weighter}
sim_funcs = {"unif": equal_weighter}

est_funcs = {"ols": get_reg_model(), "ney": neyman_estimator}


@torch.no_grad()
def test_ate():
    data_len_partial = num_imgs * len(sp_funcs) * len(sim_funcs) * len(est_funcs)
    data_len = num_reps * len(samples) * len(pert_funcs) * data_len_partial

    if asymptotic_sim:
        data_len += data_len_partial
    print(f"{asymptotic_sim=}, {data_len=}")

    data = []

    with tqdm(total=data_len) as pbar:
        for n_img, (img, label, img_path) in enumerate(dl, 1):
            if n_img > num_imgs:
                break

            if n_img % 10 == 0:
                save_data(data)

            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

            prob, label_idx = get_top_1(cnn_model, img)

            for sp_name, sp_f in sp_funcs.items():
                seg, num_segs = sp_f(img)

                if np.all(samples <= 1):  # relative
                    n_samples = ((2 ** num_segs) * samples).astype(int)
                else:
                    n_samples = samples

                if not testing:
                    if abs(num_segs - x_num * y_num) > seg_threshold:
                        continue

                    if np.any(np.array(n_samples) < min_sample):
                        continue

                DataPoint = get_DataPoint(img_path[0], sp_name, num_segs)
                get_ate_partial = partial(get_coefs, img=img, label_idx=label_idx, seg=seg, sim_funcs=sim_funcs,
                                          est_funcs=est_funcs, datapoint_class=DataPoint, data=data, pbar=pbar)

                print(f"{n_img=}, {sp_name=}, {num_segs=}, {n_samples=}, {label=}, {label_idx=}, {prob=}")
                if asymptotic_sim:
                    asym_coefs_dict = nested_dict()
                    ate = get_ate_partial(ate=None, num_reps=1, n_samples=(2 ** num_segs,),
                                          pert_funcs={"perm": permutation_gen_batched},
                                          is_asymptotic=True, asym_coefs_dict=asym_coefs_dict)

                else:
                    asym_coefs_dict = None
                    ate = None

                ate = get_ate_partial(ate=ate, num_reps=num_reps, n_samples=n_samples, pert_funcs=pert_funcs,
                                      is_asymptotic=False, asym_coefs_dict=asym_coefs_dict)

                print(f"{ate.accum_samples=}, {ate.reused_samples=}")

    if testing:
        assert len(data) == data_len

    save_data(data)


def get_coefs(img, label_idx, seg, ate, num_reps, n_samples, pert_funcs, sim_funcs, est_funcs, datapoint_class,
              data, is_asymptotic, asym_coefs_dict, pbar) -> ATE:
    for n_rep in range(num_reps):
        for n_sam in n_samples:
            for pert_name, pert_f in pert_funcs.items():
                if ate is not None:
                    if not testing and ate.is_asymptotic and pert_name == 'bern':
                        ate.perturb_func_ = (pert_name, n_sam)
                    else:
                        ate.perturb_func_ = (pert_f, n_sam)
                for sim_name, sim_f in sim_funcs.items():
                    if ate is not None:
                        ate.similarity_func_ = (sim_f, False)  # don't recalculate everything
                    for est_name, est_f in est_funcs.items():
                        print(f"{n_rep=}, {n_sam=}, {pert_name=}, {sim_name=}, {est_name=}")

                        if testing or ate is None:
                            new_ate = ATE(
                                cnn_model,
                                interpretable_model=est_f,
                                similarity_func=sim_f,
                                similarity_func_multiple=sim_funcs.values(),
                                perturb_func=pert_f
                            )

                            new_coefs, new_attrs = new_ate.attribute(
                                img,
                                target=label_idx,
                                feature_mask=seg,
                                n_samples=n_sam,
                                perturbations_per_eval=batch_size,
                                return_both=True,
                                show_progress=False
                            )

                        if ate is None:
                            ate = new_ate
                            coefs = new_coefs
                            attrs = new_attrs

                        else:
                            ate.interpretable_model_ = est_f

                            coefs, attrs = ate.attribute(
                                use_previous_results=True,
                                return_both=True)

                            if testing:
                                assert torch.allclose(coefs, new_coefs, atol=1e-5, rtol=1e-2)

                        iauc, *_ = ate.get_auc('insertion')
                        dauc, *_ = ate.get_auc('deletion')

                        datapoint = datapoint_class(n_rep=n_rep,
                                                    n_sam=n_sam,
                                                    pert_name=pert_name,
                                                    sim_name=sim_name,
                                                    est_name=est_name,
                                                    coefs=coefs.squeeze(0).cpu().numpy(),
                                                    iauc=iauc,
                                                    dauc=dauc)

                        if is_asymptotic:
                            assert ate.is_asymptotic

                            asym_coefs_dict[sim_name][est_name] = coefs.squeeze(0).cpu().numpy()
                        else:
                            if asym_coefs_dict is not None:
                                print(f"{ate.prev_num_samples=}, {ate.prev_reused_samples=}")
                                assert ate.prev_num_samples == ate.prev_reused_samples

                        if asym_coefs_dict is not None:
                            datapoint.asym_coefs = asym_coefs_dict[sim_name][est_name]
                            datapoint.get_mses()

                        data.append(datapoint)
                        pbar.update()

    return ate


def get_DataPoint(img_path_, sp_name_, num_segs_):
    @dataclass()
    class DataPoint:
        n_rep: int
        n_sam: int

        pert_name: str
        sim_name: str
        est_name: str

        coefs: np.ndarray
        iauc: float
        dauc: float

        img_path: str = img_path_
        sp_name: str = sp_name_
        num_segs: int = num_segs_

        asym_coefs: np.ndarray = np.nan
        mse: float = np.nan
        rel_mse: float = np.nan

        def get_mses(self):
            assert isinstance(self.asym_coefs, np.ndarray) and np.isnan(self.rel_mse)
            self.mse = np.linalg.norm(self.coefs - self.asym_coefs) ** 2
            self.rel_mse = (np.linalg.norm(self.coefs - self.asym_coefs)
                            / np.linalg.norm(self.asym_coefs)) ** 2

    return DataPoint


def save_data(data):
    df = pd.DataFrame(data)

    if not asymptotic_sim:
        assert df["asym_coefs"].isna().all()
        assert df["rel_mse"].isna().all()

    save_name = f"saliency_{asymptotic_sim=}_" \
                f"{socket.gethostname().split('.')[0]}_{device}_" \
                f"{y_num=}_{x_num=}_{num_imgs=}_{num_reps=}_" \
                f"{datetime.now().strftime('%y.%m.%d.%H.%M.%S')}.feather"

    df.to_feather(os.path.join(data_path, save_name))

    if testing:
        print(len(data))
        print(data)

        print(df.groupby(["pert_name", "sim_name", "est_name"])["rel_mse"].mean())
        print(df)
