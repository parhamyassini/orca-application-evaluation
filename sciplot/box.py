from . import AXIS_FONT_SIZE, TICK_FONT_SIZE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import utils

__author__ = 'Khaled Diab (kdiab@sfu.ca)'


def plot_box(xs, ys,
             data,
             xaxis_label=None, yaxis_label=None,
             x_sci=False, y_sci=True,
             name=None):
    fig, ax = plt.subplots()
    ax = sns.boxplot(x=xs, y=ys, data=data, linewidth=1, fliersize=6)
    utils.set_sci_axis(ax, x_sci, y_sci)
    utils.set_axis_labels(ax, xaxis_label, yaxis_label)
    utils.finalize(ax)
    utils.save_fig(name)