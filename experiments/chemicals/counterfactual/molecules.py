from functools import partial
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from tqdm import trange

RANDOM_SEED = 62

SAVE_DIR = Path("psiturk/static/stimuli")
SAVE_DIR.mkdir(exist_ok=True)

NUM_IMG = 50  # 30 main trials + some demo trials
CONCAT_IMG = False  # set to False in production

FIG_SIZE = 8
FIG_DPI = 100  # contour only

AX_START = -1
AX_END = 1

GRAPH_RANGE = 0.75  # graphviz radius

# chemical attributes
MIN_ATM = 4  # per molecule
MAX_ATM = 6

MIN_REP = 2  # num repeats
MAX_REP = 3

MIN_MOL = 2  # num molecule
MAX_MOL = 2

# matplotlib colors
BG_COLOR = '.25'
OCC_COLOR = '.75'

EDGE_COLOR = 'w'  # white
NODE_EDGE_COLOR = 'k'  # black
NODE_COLOR = sns.color_palette("colorblind")
NODE_SIZES = np.linspace(100, 150, 10)
NODE_SHAPES = np.array(list('so^>v<dph8'))

SAL_PALETTE = "flare_r"
SAL_BW_ADJUST = 0.5
SAL_THRESH = 0.25
SAL_LEVELS = 100

rng = np.random.default_rng(RANDOM_SEED)
NODE_COLOR.pop(7)  # remove gray

sample_int = partial(rng.integers, endpoint=True)
sample_choice = partial(rng.choice, replace=False)

# get eligible molecules
# Ref: https://networkx.org/documentation/stable/auto_examples/graphviz_layout/plot_atlas.html#sphx-glr-auto-examples-graphviz-layout-plot-atlas-py
atlas = nx.graph_atlas_g()
all_mols = []

for g in atlas:
    if nx.number_connected_components(g) == 1:
        if MAX_ATM >= g.number_of_nodes() >= MIN_ATM:
            if not any(nx.is_isomorphic(g, m) for m in all_mols):
                all_mols.append(g)

# When MIN_ATM == MAX_ATM
# using all_mols = np.array(all_mols, dtype="object")
# will convert to int array
arr_tmp = np.empty(len(all_mols), dtype="object")
for i in range(len(all_mols)):
    arr_tmp[i] = all_mols[i]
all_mols = arr_tmp

for n_img in trange(NUM_IMG):
    # molecules
    n_mol = sample_int(MIN_MOL, MAX_MOL)
    sampled_mol = sample_choice(all_mols, n_mol)

    # salient
    n_sal = sample_int(1, n_mol - 1)
    for c in sampled_mol[:n_sal]:
        nx.set_node_attributes(c, True, "sal")

    for c in sampled_mol[n_sal:]:
        nx.set_node_attributes(c, False, "sal")

    # colors and shapes per molecule
    sampled_colors = sample_choice(NODE_COLOR, n_mol)
    sampled_shapes = sample_choice(NODE_SHAPES, n_mol)

    rep_mols = []
    for g, cl, sh in zip(sampled_mol, sampled_colors, sampled_shapes):
        nx.set_node_attributes(g, cl, "plt_color")
        nx.set_node_attributes(g, sh, "plt_shape")

        # sizes per atom
        sampled_sizes = sample_choice(NODE_SIZES, g.number_of_nodes())
        nx.set_node_attributes(g, dict(zip(g.nodes, sampled_sizes)), "plt_size")

        # repeat molecules
        rep_mols.extend(repeat(g, sample_int(MIN_REP, MAX_REP)))

    # combine
    mol_graph = nx.disjoint_union_all(rep_mols)

    # visualize
    # https://www.graphviz.org/pdf/neatoguide.pdf
    # https://graphviz.org/pdf/dot.1.pdf
    mol_pos = nx.nx_agraph.graphviz_layout(mol_graph, prog="neato", args="-Gmode=KK")
    mol_pos = nx.rescale_layout_dict(mol_pos, GRAPH_RANGE)

    sal_mol = []
    nonsal_mol = []

    # gather salient and non-salient molecules
    for g in (mol_graph.subgraph(c) for c in nx.connected_components(mol_graph)):
        if all(list(nx.get_node_attributes(g, "sal").values())):
            sal_mol.append(g)
        else:
            nonsal_mol.append(g)

    if CONCAT_IMG:
        fig, ((ax_mol, ax_sal), (ax_opp, ax_same)) = plt.subplots(2, 2, figsize=(2 * FIG_SIZE, 2 * FIG_SIZE))
    else:
        fig_mol, ax_mol = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
        fig_sal, ax_sal = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
        fig_same, ax_same = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))  # retain salient molecules
        fig_opp, ax_opp = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))  # conceal salient molecules

    # graph molecules
    for axis, mols in zip((ax_mol, ax_sal, ax_opp, ax_same),
                          (sal_mol + nonsal_mol, sal_mol + nonsal_mol, nonsal_mol, sal_mol)):

        axis.set_xlim(AX_START, AX_END)
        axis.set_ylim(AX_START, AX_END)

        for m in mols:
            shape = list(nx.get_node_attributes(m, "plt_shape").values())[0]  # one shape per molecule
            colors = list(nx.get_node_attributes(m, "plt_color").values())
            sizes = list(nx.get_node_attributes(m, "plt_size").values())

            nx.draw(m, mol_pos, node_size=sizes, node_color=colors, node_shape=shape,
                    edge_color=EDGE_COLOR, edgecolors=NODE_EDGE_COLOR, with_labels=False, ax=axis)

    # graph saliency map
    sal_pos = [mol_pos[n] for m in sal_mol for n in m.nodes]
    sal_pos = list(map(list, zip(*sal_pos)))  # transpose

    sns.kdeplot(x=sal_pos[0], y=sal_pos[1], ax=ax_sal, fill=True,
                levels=SAL_LEVELS, thresh=SAL_THRESH, bw_adjust=SAL_BW_ADJUST,
                cmap=SAL_PALETTE, zorder=-10)

    ax_sal.set_rasterization_zorder(0)  # rasterise contour

    if CONCAT_IMG:
        fig.savefig(SAVE_DIR / f'fig_{n_img}.svg', facecolor=BG_COLOR, bbox_inches='tight', dpi=FIG_DPI)
    else:
        fig_mol.savefig(SAVE_DIR / f'ax_mol_{n_img}.svg', facecolor=BG_COLOR, bbox_inches='tight')
        fig_sal.savefig(SAVE_DIR / f'ax_sal_{n_img}.svg', facecolor=BG_COLOR, bbox_inches='tight')
        fig_opp.savefig(SAVE_DIR / f'ax_opp_{n_img}.svg', facecolor=BG_COLOR, bbox_inches='tight')
        fig_same.savefig(SAVE_DIR / f'ax_same_{n_img}.svg', facecolor=BG_COLOR, bbox_inches='tight')

    plt.close('all')
