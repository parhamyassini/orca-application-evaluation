import itertools
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from . import utils

__author__ = 'Khaled Diab (kdiab@sfu.ca)'

BAR_PALETTE = list(reversed(['#ffffcc', '#c2e699', '#78c679', '#31a354', '#006837']))

# FONT_FAMILY = 'Times New Roman'
FONT_FAMILY = 'Linux Libertine O'

# FONT_DICT = {'family': 'serif', 'serif': 'Times New Roman'}
FONT_DICT = {'family': FONT_FAMILY}


def plot_n_bars(ys,
                labels=None,
                yerrs=None,
                legend=True,
                legend_top=True,
                legend_font_size=28,
                legend_bbox=None,
                xaxis_label=None,
                yaxis_label=None,
                xticks=None, xticks_kwargs=None, xticks_params=None, xticks_labels=None,
                yticks=None, yticks_kwargs=None, yticks_params=None,
                y_lim=None, xtick_step=1, bar_width=0.2,
                x_sci=False, y_sci=False,
                hatch=False,
                lines=None,
                lines_kw=None,
                arrows=None,
                x_grid=True,
                y_grid=True,
                hlines_dict=None,
                fig_size=None,
                stacked=False,
                secondary=False,
                secondary_x=None,
                secondary_y=None,
                secondary_ylim=None,
                secondary_y_label=None,
                secondary_yticks=None,
                name=None):
    # with BAR_PALETTE:
    plt.rc('font', **FONT_DICT)
    plt.rc('ps', **{'fonttype': 42})
    plt.rc('pdf', **{'fonttype': 42})
    plt.rc('mathtext', **{'fontset': 'cm'})
    plt.rc('ps', **{'fonttype': 42})
    plt.rc('legend', handlelength=1., handletextpad=0.1)

    ax2 = None
    fig, ax = plt.subplots()

    if fig_size and isinstance(fig_size, list) and len(fig_size) > 0:
        if len(fig_size) == 1:
            fig.set_figwidth(fig_size[0])
        else:
            fig.set_figwidth(fig_size[0])
            fig.set_figheight(fig_size[1])

    n = len(ys)
    x = range(1, xtick_step * len(ys[0]) + 1, xtick_step)

    xs = [[]]*n
    rects = []
    for i in range(n):
        if not stacked:
            if n % 2 == 0:
                if i < n / 2:
                    xs[i] = [x_ - (n / 2 - i) * bar_width for x_ in x]
                else:
                    xs[i] = [x_ + (i - (n / 2)) * bar_width for x_ in x]
            else:
                if i < n / 2:
                    xs[i] = [x_ - (int(n / 2) - i) * bar_width for x_ in x]
                elif i > n / 2:
                    xs[i] = [x_ + (i - int(n / 2)) * bar_width for x_ in x]
                elif i == int(n / 2):
                    xs[i] = [x_ for x_ in x]
        else:
            xs[i] = [x_ for x_ in x]

    # colors = utils.get_color_cycler(colors=BAR_PALETTE)
    # print(colors)
    c_idx = 0
    colors = itertools.cycle(BAR_PALETTE)
    hatches = utils.get_hatch_cycler()
    np_y = []
    for i in range(n):
        np_y.append(np.array(ys[i]))

    for i in range(n):
        bar_args = {}
        if hatch:
            h = next(hatches)
            bar_args['hatch'] = h
        c = BAR_PALETTE[c_idx]
        c_idx += 1
        if c_idx == len(BAR_PALETTE):
            c_idx = 0
        if not stacked:
            rects.append(ax.bar(xs[i], ys[i], align='center', width=bar_width, edgecolor='k', color=c, lw=0., **bar_args))
        else:
            bottom = np.zeros(np_y[0].shape)
            for j in range(0, i):
                bottom = bottom + np_y[j]
            # print(i, xs[i])
            p = ax.bar(xs[i], ys[i], align='center', width=bar_width, bottom=bottom, color=c, lw=0., **bar_args)
            rects.append(p)

    if xticks and isinstance(xticks, list):
        # print('XX')
        # new_ticks = xticks
        new_ticks = []
        for i in range(len(xs[0])):
            _sum = 0
            for _list in xs:
                _sum += _list[i]
            new_ticks.append(_sum / float(len(xs)))
        # print(new_ticks)
        if xticks_labels:
            if xticks_kwargs:
                plt.xticks([x_ for x_ in new_ticks], xticks_labels, **xticks_kwargs)
            else:
                # print('>>', [x_ for x_ in xticks])
                # print('>>', xticks_labels)
                plt.xticks([x_ for x_ in new_ticks], xticks_labels)
        else:
            if xticks_kwargs:
                plt.xticks([x_ for x_ in new_ticks], xticks, **xticks_kwargs)
            else:
                plt.xticks([x_ for x_ in new_ticks], xticks)

    if xticks_params:
        ax.tick_params(axis='x', **xticks_params)

    if y_lim and isinstance(y_lim, list) and len(y_lim) > 0:
        if len(y_lim) == 1:
            plt.ylim(bottom=y_lim[0])
        else:
            plt.ylim(bottom=y_lim[0])
            plt.ylim(top=y_lim[1])

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

    if lines:
        for line in lines:
            bar_number = line.get('bar_number', 0)
            line_x = xs[bar_number]
            line_y = ys[bar_number]
            fc = rects[bar_number].patches[0].get_fc()
            if lines_kw:
                ax.plot(line_x, line_y, color=fc, **lines_kw)
            else:
                ax.plot(line_x, line_y, color=fc)

    if legend:
        ax.legend(rects, labels)
        # bbox = (0., 0.95, 1., .10)
        # if legend_bbox:
        #     bbox = legend_bbox
        # if legend_top:
        #     ax.legend(rects, labels,
        #               fontsize=legend_font_size, frameon=False,
        #               bbox_to_anchor=bbox,
        #               handlelength=1, handletextpad=0.2,
        #               loc=3, ncol=n, mode="expand",
        #               borderaxespad=0.)
        # else:
        #     ax.legend(rects, labels,
        #               fontsize=legend_font_size, frameon=False,
        #               bbox_to_anchor=bbox,
        #               handlelength=1, handletextpad=0.2,
        #               borderaxespad=0.)
    if arrows:
        utils.draw_arrows(arrows)

    utils.set_sci_axis(ax, x_sci, y_sci)
    utils.set_axis_labels(ax, xaxis_label, yaxis_label)
    utils.finalize(ax, x_grid=x_grid, y_grid=y_grid)

    # if secondary and secondary_y:
    #     ax2 = fig.add_subplot()

    # line_color = '#0072B2'
    line_color = '#0000FF'
    if secondary and secondary_y:
        ax2 = ax.twinx()
        ax2.plot(secondary_x, secondary_y, color=line_color, ls='-', marker='o', markersize=10)
        ax2.set_ylabel(secondary_y_label, color=line_color)
        if secondary_ylim:
            ax2.set_ylim(secondary_ylim)
        
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.tick_params('y', colors=line_color)
        if secondary_yticks:
            ax2.yaxis.set_ticks(secondary_yticks)
        # ax2.yaxis.set_ticks([19., 19.2, 19.4, 19.6, 19.8, 20.])
        utils.finalize(ax2, x_grid=x_grid, y_grid=y_grid, despine_right=False)
        plt.tight_layout()

    utils.save_fig(name)
