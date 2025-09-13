from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

SAVE_DIR = Path("psiturk/static/images")
SAVE_DIR.mkdir(exist_ok=True)

SAL_PALETTE = "flare_r"

# Ref: https://matplotlib.org/stable/tutorials/colors/colorbar_only.html#customized-colorbars-tutorial
cmap = sns.color_palette(SAL_PALETTE, as_cmap=True)
fig, ax = plt.subplots(figsize=(7, 1))
fig.subplots_adjust(bottom=0.5)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=ax, orientation='horizontal', ticks=[])

# https://stackoverflow.com/questions/23922804/draw-arrow-outside-plot-in-matplotlib
cb.ax.annotate('', xy=(1, -0.3), xycoords='axes fraction', xytext=(0, -0.3),
               arrowprops=dict(facecolor='black', width=2, headwidth=8))
cb.set_label('Higher Importance', size=15, labelpad=15)
fig.savefig(SAVE_DIR / 'colorbar.svg')
