# from . import LINE_STYLES, LEGEND_FONT_SIZE
import os
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import seaborn as sns

__author__ = 'Khaled Diab (kdiab@sfu.ca)'

DEFAULT_TWO_HEAD_ARROW = {'arrowstyle': '<->', 'mutation_scale': 40., 'linewidth': 3, 'color': 'm', 'alpha': 1}
DEFAULT_ONE_HEAD_ARROW = {'arrowstyle': '->', 'mutation_scale': 40., 'linewidth': 3, 'color': 'm', 'alpha': 1}

"""
arrows is a list of an arrow dictionary
Every arrow dictionary is in the form:
arr = {'x1y1': (),
       'x2y2': (),
       'style': {'arrowstyle': '<->', 'mutation_scale': 40., 'linewidth': 3, 'color': 'm', 'alpha': 1},
       'text': '',
       'text_x1y1': (),
       'text_x2y2': (),
       'text_size': 28
       }


hlines_dict = 
{
'y': val,
'xmin': val,
'xmax': val,
}
"""

def is_list_of_list(ls):
    return all(is_list(elem) for elem in ls)


def is_list(ls):
    return isinstance(ls, list) or isinstance(ls, np.ndarray)


def set_sci_axis(ax, x_sci, y_sci):
    if x_sci and isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    if y_sci and isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter):
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


def set_axis_labels(ax, x_label, y_label):
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


def set_legend(top=True, ncol=1, font_size=24, bbox=None):
    if top:
        if ncol == 2:
            if not bbox:
                bbox = (0.1, 0.94, 0.85, .10)
            plt.legend(fontsize=font_size, frameon=False,
                       bbox_to_anchor=bbox,
                       loc=3, ncol=ncol, mode='expand',
                       borderaxespad=0.)
        else:
            if not bbox:
                bbox = (0.0, 0.94, 1, .2)
            plt.legend(fontsize=font_size, frameon=False,
                       bbox_to_anchor=bbox,
                       loc=3, ncol=ncol, mode='expand',
                       borderaxespad=0.)
    else:
        # plt.legend(fontsize=20, fancybox=False, frameon=False, bbox_to_anchor=(-0.05, 0.95, 0.35, .10), ncol=2)
        if not bbox:
            # bbox = (0.1, 0.98, 0.4, .10)
            plt.legend(fontsize=font_size, fancybox=False, frameon=True,
                    ncol=1,
                    loc=0,
                    mode='expand', borderaxespad=0., edgecolor='white', facecolor='white', framealpha=1)
        else:
            plt.legend(fontsize=font_size, fancybox=False, frameon=True,
                    bbox_to_anchor=bbox,
                    ncol=1,
                    loc=0,
                    mode='expand', borderaxespad=0., edgecolor='white', facecolor='white', framealpha=1)


def finalize(ax, tight=True, despine_left=False, despine_bottom=False, despine_right=True, despine_top=True,
             x_grid=True, y_grid=True):

    sns.despine(ax=ax, top=despine_top, right=despine_right, left=despine_left, bottom=despine_bottom)

    if x_grid:
        ax.xaxis.grid(ls='--', alpha=0.6)

    if y_grid:
        ax.yaxis.grid(ls='--', alpha=0.6)

    if tight:
        plt.tight_layout()


def get_ls(ls_cycle):
    if LINE_STYLES and is_list(LINE_STYLES) and len(LINE_STYLES) > 0:
        return LINE_STYLES if ls_cycle else LINE_STYLES[0]
    else:
        return ['-']


def get_marker_style(cycle):
    # MARKERSTYLES = ['o', 'v', '^', '<', '>']
    MARKERSTYLES = ['o', 's', 'v', '^', '<', '>']
    if MARKERSTYLES and is_list(MARKERSTYLES) and len(MARKERSTYLES) > 0:
        return MARKERSTYLES if cycle else MARKERSTYLES[0]
    else:
        return ['1']


def get_line_styles_cycler(ls_cycle):
    return cycle(get_ls(ls_cycle))


def get_marker_styles_cycler(ls_cycle):
    return cycle(get_marker_style(ls_cycle))


# def get_color_cycler(reverse=False, colors=flatui):
#     _colors = list(colors)
#     if reverse:
#         return cycle(reversed(_colors))

#     return cycle(_colors)


def get_hatch_cycler():
    patterns = ('\\\\', '//', 'o', 'X', 'O', '*')
    return cycle(patterns)


def save_fig(full_path, default_format='eps', dpi=300):
    if full_path:
        filename, file_extension = os.path.splitext(full_path)
        if '.' in file_extension:
            file_extension = file_extension[1:]
        else:
            file_extension = default_format
            full_path = full_path + '.' + file_extension

        plt.savefig(full_path, format=file_extension, dpi=dpi)


def draw_arrows(arrows):
    for arr in arrows:
        if arr:
            x1y1 = arr.get('x1y1', (0, 0))
            x2y2 = arr.get('x2y2', (0, 0))
            style = arr.get('style', DEFAULT_TWO_HEAD_ARROW)
            text = arr.get('text', '')
            text_x1y1 = arr.get('text_x1y1', (0, 0))
            text_x2y2 = arr.get('text_x2y2', (0, 0))
            text_size = arr.get('text_size', 28)
            plt.annotate('', xy=x1y1, xycoords='data',
                         xytext=x2y2, textcoords='data',
                         arrowprops=style)
            plt.annotate(text, ha='center', xy=text_x1y1, xycoords='data', xytext=text_x2y2,
                         # textcoords='data',
                         textcoords='offset points',
                         size=text_size)
