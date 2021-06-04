from copy import copy
import numpy as np


def make_background_neutral(heatmap, ref_min=None, ref_max=None):
    """make_background_neutral.
    For the sake of visibility, puts the background neutral.
     = puts the background value as the mean of the heatmap distribution.

    :param heatmap: np.ndarray.
    :param ref_min: fixed min value of the cmap, if none, fixed to min(heatmap).
    :param ref_max:fixed max value of the cmap, if none, fixed to min(heatmap).
    """
    heatmap_neutral = copy(heatmap)
    ref_min = heatmap_neutral.min() if ref_min is None else ref_min
    ref_max = heatmap_neutral.max() if ref_max is None else ref_max
    heatmap_neutral[heatmap_neutral == 0] = np.mean([ref_min, ref_max])
    return heatmap_neutral
    """
    """
 
def add_titlebox(ax, text):
    """add_titlebox.

    :param ax: matplotlib axes object
    :param text: str, texte to print at the corner bottom-left of the axes.
    """
    ax.text(.05, .05, text,
        horizontalalignment='left',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
        fontsize=20)
    return ax

def set_axes_color(ax, color='orange'):
    """set_axes_color. As its name says...

    :param ax: matplotlib axes object
    :param color: name of color, must be compatible with matplotlib colors.
    """
    dirs = ['bottom', 'top', 'left', 'right']
    args = {x:False for x in dirs}
    args.update({'label'+x:False for x in dirs})
    args.update({'axis':'both', 'which': 'both'})
    ax.tick_params(**args)
    for sp in ax.spines:
        ax.spines[sp].set_color(color)
        ax.spines[sp].set_linewidth(5)
    return ax

