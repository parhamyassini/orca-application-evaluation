from . import LEGEND_FONT_SIZE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from . import utils

__author__ = 'Khaled Diab (kdiab@sfu.ca)'


def plot_line(xs,
              ys,
              line_labels=None,
              xaxis_label=None,
              yaxis_label=None,
              xticks=None,
              xticks_labels=None,
              xticks_kwargs=None,
              xticks_params=None,
              yticks_major=None,
              yticks=None,
              yticks_labels=None,
              yticks_kwargs=None,
              yticks_params=None,
              vlines=None,
              vlines_kwargs=None,
              hlines_dict=None,
              x_sci=True,
              y_sci=True,
              y_lim=None,
              x_lim=None,
              legend=True,
              legend_top=True,
              legend_font_size=LEGEND_FONT_SIZE,
              legend_bbox=None,
              line_colors=None,
              ls_cycle=False,
              marker_size=0,
              x_log_scale=False,
              y_log_scale=False,
              arrows=None,
              x_grid=True,
              y_grid=True,
              draw_vertical_region=None,
              fig_size=None,
              name=None,
              rc=None):
    multiple_x = utils.is_list_of_list(xs)
    multiple_y = utils.is_list_of_list(ys)
    multiple_line_label = utils.is_list(line_labels)
    assert multiple_x == multiple_y == multiple_line_label

    if rc:
        sns.set_context(context='paper', rc=rc)

    fig, ax = plt.subplots()
    if fig_size and isinstance(fig_size, list) and len(fig_size) > 0:
        if len(fig_size) == 1:
            fig.set_figwidth(fig_size[0])
        else:
            fig.set_figwidth(fig_size[0])
            fig.set_figheight(fig_size[1])
    ls_cycler = utils.get_line_styles_cycler(ls_cycle)
    ms_cycler = utils.get_marker_styles_cycler(marker_size > 0)

    if multiple_x:
        for x, y, line_label in zip(xs, ys, line_labels):
            if x_log_scale:
                ax.semilogx(x, y, label=line_label, ls=next(ls_cycler),
                            marker=next(ms_cycler), markersize=marker_size)
            else:
                ax.plot(x, y, label=line_label, ls=next(ls_cycler), marker=next(ms_cycler), markersize=marker_size)
    else:
        if x_log_scale:
            ax.semilogx(xs, ys, label=line_labels, ls=next(ls_cycler), marker=next(ms_cycler), markersize=marker_size)
        else:
            ax.plot(xs, ys, label=line_labels, ls=next(ls_cycler), marker=next(ms_cycler), markersize=marker_size)

    if y_log_scale:
        ax.set_yscale('log')

    if hlines_dict:
        colors = utils.get_color_cycler(reverse=True)
        for idx, hline in enumerate(hlines_dict):
            y = hline.get('y', 0)
            xmin = hline.get('xmin', 0)
            xmax = hline.get('xmax', 0)
            line_label = hline.get('label', '')
            line_width = hline.get('lw', '1')
            if line_label:
                ax.plot([xmin, xmax], [y, y], label=line_label, ls='-', markersize=0, color=next(colors), lw=line_width)
            else:
                ax.plot([xmin, xmax], [y, y], ls='-', markersize=0, color=next(colors), lw=line_width)

    if xticks and isinstance(xticks, list) and xticks_labels and isinstance(xticks_labels, list):
        if len(xticks) == len(xticks_labels):
            # x = xs[0] if multiple_x else xs
            if xticks_kwargs:
                plt.xticks(xticks, xticks_labels, **xticks_kwargs)
            else:
                plt.xticks(xticks, xticks_labels)

    if xticks_params:
        ax.tick_params(axis='x', **xticks_params)

    # major_ticks = np.arange(0, 30, 10)
    # ax.set_yticks(major_ticks)

    if yticks_major:
        ax.set_yticks(yticks_major)

    if yticks and isinstance(yticks, list) and yticks_labels and isinstance(yticks_labels, list):
        # print(yticks)
        # print(yticks_labels)
        if len(yticks) == len(yticks_labels):
            # x = xs[0] if multiple_x else xs

            if yticks_kwargs:
                plt.yticks(yticks, yticks_labels, **yticks_kwargs)
            else:
                plt.yticks(yticks, yticks_labels)

    if yticks_params:
        ax.tick_params(axis='y', **yticks_params)

    if y_lim and isinstance(y_lim, list) and len(y_lim) > 0:
        if len(y_lim) == 1:
            plt.ylim(bottom=y_lim[0])
        else:
            plt.ylim(bottom=y_lim[0])
            plt.ylim(top=y_lim[1])

    if x_lim and isinstance(x_lim, list) and len(x_lim) > 0:
        if len(x_lim) == 1:
            plt.xlim(left=x_lim[0])
        else:
            plt.xlim(left=x_lim[0])
            plt.xlim(right=x_lim[1])

    # vlines = vlines or []
    # for xvline in vlines:
    #     with ALTERNATIVE_PALETTE:
    #         if vlines_kwargs:
    #             plt.axvline(x=xvline, **vlines_kwargs)
    #         else:
    #             plt.axvline(x=xvline)

    # hlines = hlines or []
    # for yhline in hlines:
    #     # with ALTERNATIVE_PALETTE:
    #     if hlines_kwargs:
    #         plt.axhline(y=yhline, **hlines_kwargs)
    #     else:
    #         plt.axhline(y=yhline)

    # if hlines_dict:
    #     y = hlines_dict.get('y', 0)
    #     xmin = hlines_dict.get('xmin', 0)
    #     xmax = hlines_dict.get('xmax', 0)
    #     line_label = hlines_kwargs.get('label', '')
    #     line_color = hlines_kwargs.get('color', '')
    #     line_width = hlines_kwargs.get('lw', '')
    #     ax.plot([xmin, xmax], [y, y], label=line_label, ls='-', markersize=0, color=line_color, lw=line_width)
        # if hlines_kwargs:
        #     plt.hlines(y, xmin, xmax, **hlines_kwargs)
        # else:
        #     plt.hlines(y, xmin, xmax)

    if draw_vertical_region and isinstance(draw_vertical_region, dict):
        v_region = draw_vertical_region.get('region', (0, 0))
        v_color = draw_vertical_region.get('color', 'w')
        v_alpha = draw_vertical_region.get('alpha', 0.25)
        ax.axvspan(v_region[0], v_region[1], alpha=v_alpha, color=v_color)

    ncol = len(xs) if multiple_x else 1
    if hlines_dict:
        ncol += len(hlines_dict)
    if legend:
        utils.set_legend(legend_top, ncol, font_size=legend_font_size, bbox=legend_bbox)
    utils.set_sci_axis(ax, x_sci, y_sci)
    utils.set_axis_labels(ax, xaxis_label, yaxis_label)

    if arrows:
        utils.draw_arrows(arrows)

    # if draw_arrow:
    # plt.annotate('Early-quitters', xy=(10, 0.05), xycoords='data',
    #              xytext=(15, 0.1), textcoords='data', size=18,
    #              arrowprops={'arrowstyle': '->', 'linewidth': 2, 'color': 'k', 'alpha': 1})
    #
    # plt.annotate('Drop-out', xy=(50, 0.01), xycoords='data',
    #              xytext=(40, 0.06), textcoords='data', size=18,
    #              arrowprops={'arrowstyle': '->', 'linewidth': 2, 'color': 'k', 'alpha': 1})
    #
    # plt.annotate('Steady viewers', xy=(90, 0.05), xycoords='data',
    #              xytext=(55, 0.1), textcoords='data', size=18,
    #              arrowprops={'arrowstyle': '->', 'linewidth': 2, 'color': 'k', 'alpha': 1})
    #     plt.annotate('$3$X', xy=(215, 1.5), xycoords='data', xytext=(0, 0), textcoords='offset points', size=28)
    #
    # if draw_arrow2:
    #     plt.annotate('', xy=(390, 0.5), xycoords='data',
    #                  xytext=(775, 0.5), textcoords='data',
    #                  arrowprops={'arrowstyle': '<->', 'mutation_scale': 40., 'linewidth': 3, 'color': 'm', 'alpha': 1})
    #     plt.annotate('$2$X', xy=(555, 0.52), xycoords='data', xytext=(0, 0), textcoords='offset points', size=28)
    #
    #     plt.annotate('', xy=(455, 0.97), xycoords='data',
    #                  xytext=(770, 0.97), textcoords='data',
    #                  arrowprops={'arrowstyle': '<->', 'mutation_scale': 35., 'linewidth': 3, 'color': 'm', 'alpha': 1})
    #     plt.annotate('$1.7$X', xy=(550, 0.96), xycoords='data',
    #                  xytext=(300, 0.8), textcoords='data',
    #                  size=28, va='center', ha='center',
    #                  arrowprops=dict(arrowstyle='fancy', connectionstyle='arc3,rad=0.2', color='k', alpha=1.)
    #                  )

    utils.finalize(ax, x_grid=x_grid, y_grid=y_grid)
    utils.save_fig(name)
    # if rc:
    #     sns.set_context(context='paper', rc=DEFAULT_RC)
