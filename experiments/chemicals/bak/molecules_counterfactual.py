from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

RANDOM_SEED = 0

SAVE_DIR = Path("psiturk_occlusions/static/stimuli")
SAVE_DIR.mkdir(exist_ok=True)

NUM_IMG = 5
CONCAT_IMG = True  # set to False in production

FIG_SIZE = 8
FIG_DPI = 100  # contour only

AX_START = -1
AX_END = 1

GRAPH_RANGE = 0.75  # graphviz radius

# chemical attributes
MIN_ATM = 4  # per molecule
MAX_ATM = 5

MIN_MOL = 3  # num molecule
MAX_MOL = 4

MIN_REP = 2  # num repeats
MAX_REP = 3

MIN_SAL = 1  # num salient
MAX_SAL = 2

# matplotlib colors
BG_COLOR = '.25'
OCC_COLOR = '.75'

EDGE_COLOR = 'w'  # white
NODE_EDGE_COLOR = 'k'  # black
NODE_COLOR = sns.color_palette("colorblind")
NODE_SIZES = np.linspace(75, 100, 10)
NODE_SHAPES = np.array(list('so^>v<dph8'))

SAL_PALETTE = "flare_r"
SAL_BW_ADJUST = .5
SAL_THRESH = 0.25
SAL_LEVELS = 100

rng = np.random.default_rng(RANDOM_SEED)

# get eligible molecules
# Ref: https://networkx.org/documentation/stable/auto_examples/graphviz_layout/plot_atlas.html#sphx-glr-auto-examples-graphviz-layout-plot-atlas-py
atlas = nx.graph_atlas_g()
all_mols = []

for g in atlas:
    if nx.number_connected_components(g) == 1:
        if MAX_ATM >= g.number_of_nodes() >= MIN_ATM:
            if not any(nx.is_isomorphic(g, m) for m in all_mols):
                all_mols.append(g)

all_mols = np.array(all_mols, dtype="object")
print(f"{len(all_mols)=}")

for n_img in range(NUM_IMG):
    print(f"{n_img=}")

    # molecules
    n_mol = rng.integers(MIN_MOL, MAX_MOL, endpoint=True)
    print(f"{n_mol=}")
    sampled_mol = rng.choice(all_mols, n_mol, replace=False)

    # salient
    n_sal = rng.integers(MIN_SAL, MAX_SAL, endpoint=True)
    print(f"{n_sal=}")
    rng.shuffle(sampled_mol)
    for c in sampled_mol[:n_sal]:
        nx.set_node_attributes(c, True, "sal")

    for c in sampled_mol[n_sal:]:
        nx.set_node_attributes(c, False, "sal")

    # colors and shapes per molecule
    sampled_colors = rng.choice(NODE_COLOR, n_mol, replace=False)
    sampled_shapes = rng.choice(NODE_SHAPES, n_mol, replace=False)

    rep_mols = []
    for g, col, sha in zip(sampled_mol, sampled_colors, sampled_shapes):
        nx.set_node_attributes(g, col, "plt_color")
        nx.set_node_attributes(g, sha, "plt_shape")

        # sizes per atom
        sampled_sizes = rng.choice(NODE_SIZES, g.number_of_nodes(), replace=False)
        rng.shuffle(sampled_sizes)
        nx.set_node_attributes(g, dict(zip(g.nodes, sampled_sizes)), "plt_size")

        # repeat molecules
        rep_mols.extend(repeat(g, rng.integers(MIN_REP, MAX_REP, endpoint=True)))

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
        elif not any(list(nx.get_node_attributes(g, "sal").values())):
            nonsal_mol.append(g)
        else:
            raise RuntimeError

    print(f"{len(sal_mol)=}")
    print(f"{len(nonsal_mol)=}")

    if CONCAT_IMG:
        fig, ((ax_mol, ax_sal), (ax_same, ax_opp)) = plt.subplots(2, 2, figsize=(2 * FIG_SIZE, 2 * FIG_SIZE))
    else:
        fig_mol, ax_mol = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
        fig_sal, ax_sal = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
        fig_same, ax_same = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))  # retain salient molecules
        fig_opp, ax_opp = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))  # conceal salient molecules

    # graph molecules
    for ax, mols in zip((ax_mol, ax_sal, ax_opp, ax_same),
                        (sal_mol + nonsal_mol, sal_mol + nonsal_mol, nonsal_mol, sal_mol)):
        ax.set_xlim(AX_START, AX_END)
        ax.set_ylim(AX_START, AX_END)

        for m in mols:
            shape = list(nx.get_node_attributes(m, "plt_shape").values())[0]  # one shape per molecule
            colors = list(nx.get_node_attributes(m, "plt_color").values())
            sizes = list(nx.get_node_attributes(m, "plt_size").values())

            nx.draw(m, mol_pos, node_size=sizes, node_color=colors, node_shape=shape,
                    edge_color=EDGE_COLOR, edgecolors=NODE_EDGE_COLOR, with_labels=False, ax=ax)

    # graph saliency map
    sal_pos = [mol_pos[n] for m in sal_mol for n in m.nodes]
    sal_pos = list(map(list, zip(*sal_pos)))  # transpose

    sns.kdeplot(x=sal_pos[0], y=sal_pos[1], ax=ax_sal, fill=True,
                levels=SAL_LEVELS, thresh=SAL_THRESH, bw_adjust=1 / n_sal,
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
