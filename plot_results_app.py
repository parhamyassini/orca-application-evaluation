import os
import socket
import sys,os,time
import collections
import struct
import pickle
import csv
import random
import string
from multiprocessing import Process, Queue, Value, Array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
import seaborn as sns

BAR_PLOT_TICKS = 6

# Line Styles
DEFAULT_LINE_WIDTH = 4
ALTERNATIVE_LINE_WIDTH = 5
SMALL_LINE_WIDTH = 2
LINE_STYLES = ['-', '--', '-.', ':']

# Font
TEX_ENABLED = False
TICK_FONT_SIZE = 24
AXIS_FONT_SIZE = 24
LEGEND_FONT_SIZE = 22

FONT_DICT = {'family': 'serif', 'serif': 'Times New Roman'}

flatui = ["#0072B2", "#D55E00", "#009E73", "#3498db", "#CC79A7", "#F0E442", "#56B4E9"]

DEFAULT_RC = {'lines.linewidth': DEFAULT_LINE_WIDTH,
              'axes.labelsize': AXIS_FONT_SIZE,
              'xtick.labelsize': TICK_FONT_SIZE,
              'ytick.labelsize': TICK_FONT_SIZE,
              'legend.fontsize': LEGEND_FONT_SIZE,
              'text.usetex': TEX_ENABLED,
              # 'ps.useafm': True,
              # 'ps.use14corefonts': True,
              # 'font.family': 'sans-serif',
              # 'font.serif': ['Helvetica'],  # use latex default serif font
              }

sns.set_context(context='paper', rc=DEFAULT_RC)
sns.set_style(style='ticks')
plt.rc('font', **FONT_DICT)
plt.rc('ps', **{'fonttype': 42})
plt.rc('pdf', **{'fonttype': 42})
plt.rc('mathtext', **{'fontset': 'cm'})
plt.rc('ps', **{'fonttype': 42})
plt.rc('legend', handlelength=1., handletextpad=0.1)



workloads = ['broadcast_files_0', 'broadcast_files_1', 'broadcast_files_2', 'broadcast_files_3', 'broadcast_files_4']
workload_sizes = ['88', '176', '352', '704', '1408']
udp_experiments = ['n4m1', 'n8m1', 'n12m1']
experiments_sizes = [4, 8, 12]
large_files = ['broadcast_12', 'broadcast_15', 'broadcast_18', 'broadcast_21', 'broadcast_24', 'broadcast_27', 'broadcast_30', 'broadcast_33', 'broadcast_6', 'broadcast_9']
color_pallete = ['#0071b2', '#009e74', '#cc79a7', '#d54300']
color_orca = '#e69d00'
color_orca_c = 'tab:purple'

width = 0.35

def best_fit(X, Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

def autolabel(ax, rects_in):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects_in:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

def plot_app_times(path, num_receivers):
    # TODO modify here
    sender_runtime_udp_4 = []
    sender_runtime_udp_2 = []
    sender_runtime_udp_1 = []
    sender_runtime_orca = []
    for i in range (len(workloads)):
        with open(path + '/ramses/out_udp_n' + str(num_receivers) + 'm1_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_udp_1.append(float(row[1]) * 1000)
        with open(path + '/ramses/out_orca_n' + str(num_receivers) + '_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_orca.append(float(row[1]) * 1000)
   
    ind = np.arange(len(sender_runtime_orca))  # the x locations for the groups
    width = 0.45       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, sender_runtime_udp_1, width, color=color_pallete[0], edgecolor='black', hatch='//', lw=0)
    #rects2 = ax.bar(ind + width, sender_runtime_udp_2, width)
    #rects3 = ax.bar(ind + 2*width, sender_runtime_udp_4, width)
    rects4 = ax.bar(ind + width, sender_runtime_orca, width, color=color_orca, edgecolor='black', hatch='xx', lw=0)
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Running Time (ms)')
    ax.set_xlabel('Workload Size (MB)')
    ax.yaxis.set_tick_params()
    # yticks = list(np.linspace(0, 15000, 4))
    plt.ylim(bottom=0, top=15000)
    #ax.set_title('Application running time with ' + str(num_receivers) + ' receivers varying the file size loads', pad=20)
    ax.set_xticks(ind + 0.5*width)
    ax.set_xticklabels(workload_sizes)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    #upper_bound = max(sender_runtime_udp_1)
    #tick_difference = (upper_bound/BAR_PLOT_TICKS)//100*100+100
    #ax.set_yticks(np.arange(0, 8000, 1000))
    ax.legend((rects1[0], rects4[0]), ('Unicast', 'Orca'))

    ax.get_yaxis().get_offset_text().set_visible(False)
    ax_max = max(ax.get_yticks())
    exponent_axis = np.floor(np.log10(ax_max)).astype(int)
    ax.annotate(r'$\times$10$^{\mathregular{%i}}$'%(exponent_axis-1), xy=(.01, .98), xycoords='axes fraction', fontsize='xx-large', fontfamily='serif')

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                    '%d' % int(height),
                    ha='center', va='bottom', fontsize=LEGEND_FONT_SIZE - 6)

    autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    autolabel(rects4)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.yaxis.grid()
    plt.tight_layout()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.savefig('../app_runtime_n' + str(num_receivers) + 'm1.eps', ext='eps', bbox_inches="tight")
    # plt.show()

def plot_app_time_compare(path, num_receivers):
    sender_runtime_udp = []
    sender_runtime_orca = []
    sender_runtime_orca_c = []
    if num_receivers == 12:
        sender_prcos = [1, 2, 3, 6]
    elif num_receivers == 8:
        sender_prcos = [1, 2, 4, 8]
    else:
        print('Experiment names not defined, add array in {line 105}')
        exit(1)
    num_experiments = int(math.log(num_receivers, 2) + 1)
    rects = []
    for j in range (num_experiments):
        udp_new = []
        for i in range (len(workloads)):
            with open(path + '/ramses/out_udp_n' + str(num_receivers) + 'm' + str(sender_prcos[j]) + '_' + workloads[i] + '.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        udp_new.append(float(row[1]) * 1000)
        #print (udp_new)
        sender_runtime_udp.append(udp_new)
    print (sender_runtime_udp)
    
    for i in range (len(workloads)):  
        with open(path + '/ramses/out_orca_n' + str(num_receivers) + '_' + workloads[i] + '.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_orca.append(float(row[1]) * 1000)
    for i in range (len(workloads)):  
        with open(path + '/ramses/out_orca_c_n' + str(num_receivers) + '_' + workloads[i] + '.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_orca_c.append(float(row[1]) * 1000)
    ind = np.arange(len(sender_runtime_orca))  # the x locations for the groups
    width = 0.16       # the width of the bars

    fig, ax = plt.subplots(figsize=(28, 12))

    for exp_idx in range(num_experiments):
        rect = ax.bar(ind + exp_idx*width, sender_runtime_udp[exp_idx], width, color=color_pallete[exp_idx])
        rects.append(rect)
    rect_orca_c = ax.bar(ind + (num_experiments)*width, sender_runtime_orca_c, width, color=color_orca_c)
    rect_orca = ax.bar(ind + (num_experiments+1)*width, sender_runtime_orca, width, color=color_orca)
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Running Time (ms)')
    ax.set_xlabel('Total File Size (MB)')
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title('Application running time with ' + str(num_receivers) +' receivers varying the file size loads', pad=20)
    ax.set_xticks(ind + 2*width)
    ax.set_xticklabels(workload_sizes)
    if num_receivers == 4:
        ax.legend((rects[0][0], rects[1][0], rects[2][0], rect_orca), 
            ('Unicast: 1 core for every 4 destinations',
             'Unicast: 1 core for every 2 destinations',
             'Unicast: 1 core for every destinations',
             'Orca'))
    if num_receivers == 8:
        ax.legend((rects[0][0], rects[1][0], rects[2][0], rects[3][0], rect_orca_c, rect_orca), 
        ('Unicast: 1 core for every 8 destinations',
         'Unicast: 1 core for every 4 destinations',
         'Unicast: 1 core for every 2 destinations',
         'Unicast: 1 core for every destinations',
         'Orca constant 3 cores for destination machine',
         'Orca non-contrained destination machine'))
    if num_receivers == 12:
        ax.legend((rects[0][0], rects[1][0], rects[2][0], rects[3][0], rect_orca_c, rect_orca), 
        ('Unicast: 1 core for every 12 destinations',
         'Unicast: 1 core for every 6 destinations',
         'Unicast: 1 core for every 4 destinations',
         'Unicast: 1 core for every 2 destinations',
         'Orca constant 3 cores for destination machine',
         'Orca non-contrained destination machine'))
    for rect in rects: 
        autolabel(ax, rect)
    autolabel(ax, rect_orca)
    autolabel(ax, rect_orca_c)
    plt.savefig('../app_runtime_c_n'+ str(num_receivers) +'_compare.png')
    # plt.show()

def plot_app_times_complete(path, num_receiver_arr):
    ax_text_base = ['a) ', 'b) ', 'c) ', 'd) ']
    fig, ax = plt.subplots(1,3, figsize=(21, 7))
    pos_x = 0
    for num_receivers in num_receiver_arr:
        sender_runtime_udp_4 = []
        sender_runtime_udp_2 = []
        sender_runtime_udp_1 = []
        sender_runtime_orca = []
        for i in range (len(workloads)):
            with open(path + '/ramses/out_udp_n' + str(num_receivers) + 'm1_' + workloads[i] + '.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_udp_1.append(float(row[1]) * 1000)
            with open(path + '/ramses/out_orca_n' + str(num_receivers) + '_' + workloads[i] + '.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_orca.append(float(row[1]) * 1000)

        ind = np.arange(len(sender_runtime_orca))  # the x locations for the groups
        width = 0.45       # the width of the bars
        #fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax[pos_x].bar(ind, sender_runtime_udp_1, width, color=color_pallete[0], label='$Unicast$')
        #rects2 = ax.bar(ind + width, sender_runtime_udp_2, width)
        #rects3 = ax.bar(ind + 2*width, sender_runtime_udp_4, width)
        rects4 = ax[pos_x].bar(ind + width, sender_runtime_orca, width, color=color_orca, label='$Orca$')
    
        # add some text for labels, title and axes ticks
        #ax[pos_x, 1].set_ylabel('Running Time (ms)')
        #ax[pos_x, 1].set_xlabel('Total File Size (MB)')
        txt =  ax_text_base[pos_x] + str(num_receivers) + ' receiver nodes'
        ax[pos_x].text(0.5,-0.12, txt, size=12, ha="center", transform=ax[pos_x].transAxes)
        ax[pos_x].set_xticks(ind + 0.5*width)
        ax[pos_x].set_xticklabels(workload_sizes)
        #ax[pos_x, 1].legend((rects1[0], rects4[0]), 
        #    ('Unicast',
        #     'Orca'))

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax[pos_x].text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%d' % int(height),
                        ha='center', va='bottom')

        autolabel(rects1)
        #autolabel(rects2)
        #autolabel(rects3)
        autolabel(rects4)
        handles, labels = ax[pos_x].get_legend_handles_labels()
        pos_x += 1
    for x in ax.flat:
        x.set_xlabel('Total File Size (MB)')
        x.set_ylabel('Running Time (ms)')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.29,
                    wspace=0.35)
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.savefig('../app_runtime_complete_'+ 'm1.svg', ext='svg', bbox_inches="tight")
    # plt.show()

def plot_cdf(path, num_receivers, num_send_procs, goal="send_times", mode="linear"):
    file_name_udp = 'out_udp_n'+ str(num_receivers) +'m' + str(num_send_procs) + '_'
    file_name_orca = 'out_orca_n' + str(num_receivers) + '_'
    file_name_orca_c = 'out_orca_c_n' + str(num_receivers) + '_'
    base_string_udp = path + '/ramses/' + file_name_udp
    base_string_orca = path + '/ramses/' + file_name_orca
    base_string_orca_c = path + '/ramses/' + file_name_orca_c
    if goal == "wait_times":
        base_string_udp = path + '/55/container_1/' + file_name_udp
        base_string_orca = path + '/55/container_1/' + file_name_orca
        base_string_orca_c = path + '/55/container_1/' + file_name_orca_c
    fig, ax = plt.subplots(1, 1)
    pos_x = 0
    pos_y = 0
    workload_idx = 4
    for i in range (workload_idx-1, workload_idx):
        udp_times = []
        orca_times = []
        orca_times_c = []
        with open(base_string_udp + workloads[i+1] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('broadcast'):
                    udp_times.append(float(row[1]))
        with open(base_string_orca + workloads[i+1] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('broadcast'):
                    orca_times.append(float(row[1]))
        # with open(base_string_orca_c + workloads[i+1] + '.csv') as csvfile:
        #     results = csv.reader(csvfile, delimiter=',')
        #     for row in results:
        #         if row[0].startswith('broadcast'):
        #             orca_times_c.append(float(row[1]))

        udp_times_sorted = np.sort(udp_times) * 1000
        orca_times_sorted = np.sort(orca_times) * 1000
        orca_times_sorted_c = np.sort(orca_times_c) * 1000
        p_udp = 1. * np.arange(len(udp_times)) / (len(udp_times) - 1)
        p_orca = 1. * np.arange(len(orca_times)) / (len(orca_times) - 1)
        p_orca_c = 1. * np.arange(len(orca_times_c)) / (len(orca_times_c) - 1)
    #ax.set_color_cycle(['red', 'green'])
        print(pos_x%2, pos_y%2)
        ax.set_ylim(top=1)
        #title = 'CDF of ' + goal.replace("_", " ") + ' with ' + str(num_receivers) + ' receivers(total ' + workload_sizes[i+1] + 'M files)'
        #ax.set_title(title)
        ax.plot(udp_times_sorted, p_udp, label='$Unicast$',  linewidth=2.5, color=color_pallete[0])
        ax.plot(orca_times_sorted, p_orca, label='$Orca$',  linewidth=2.5, color=color_orca)
        #ax.plot(orca_times_sorted_c, p_orca_c, label='Orca Constrained Receiver',  linewidth=2, color=color_pallete[1])
        if mode == 'log':
            ax.set_xscale('log')
        #ax[pos_x%2, pos_y%2].legend(loc='best')
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        ax.yaxis.set_tick_params(labelsize=16)
        ax.xaxis.set_tick_params(labelsize=16)
        pos_y += 1
        pos_x += pos_y +1
    #for x in ax.flat:
        ax.set_xlabel(goal.replace("_", " ").capitalize() + ' (ms)', fontsize=17)
        ax.set_ylabel('CDF', fontsize=17)

    #plt.legend(loc='best')
    plt.grid(True)
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.10, right=0.95, hspace=0.29,
                    wspace=0.35)
    ax.legend(handles, labels, fontsize=14)
    if mode == 'log':
        savepath = '../' + goal + '_n' + str(num_receivers) +'m' + str(num_send_procs) + '_' + workloads[workload_idx] + '_log.eps'
    else: 
        savepath = '../' + goal + '_n' + str(num_receivers) +'m' + str(num_send_procs) + '_' + workloads[workload_idx] +'_.eps'
    plt.savefig(savepath, ext='eps', bbox_inches="tight")
    # plt.show(fig)


def plot_cdf_both(path, num_receivers, num_send_procs, mode="linear", workload_idx=4):
    file_name_udp = 'out_udp_n'+ str(num_receivers) +'m' + str(num_send_procs) + '_'
    file_name_orca = 'out_orca_n' + str(num_receivers) + '_'
    file_name_orca_c = 'out_orca_c_n' + str(num_receivers) + '_'
    send_base_string_udp = path + '/ramses/' + file_name_udp
    send_base_string_orca = path + '/ramses/' + file_name_orca
    send_base_string_orca_c = path + '/ramses/' + file_name_orca_c
    
    wait_base_string_udp = path + '/55/container_1/' + file_name_udp
    wait_base_string_orca = path + '/55/container_1/' + file_name_orca
    wait_base_string_orca_c = path + '/55/container_1/' + file_name_orca_c
    fig, ax = plt.subplots(1, 1)
    pos_x = 0
    pos_y = 0
    
    for i in range (workload_idx-1, workload_idx):
        send_udp_times = []
        wait_udp_times = []
        send_orca_times = []
        wait_orca_times = []
        send_orca_times_c = []
        wait_orca_times_c = []
        with open(wait_base_string_udp + workloads[i+1] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('broadcast'):
                    wait_udp_times.append(float(row[1]))
        with open(wait_base_string_orca + workloads[i+1] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('broadcast'):
                    wait_orca_times.append(float(row[1]))
        with open(send_base_string_orca + workloads[i+1] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('broadcast'):
                    send_orca_times.append(float(row[1]))
        with open(send_base_string_udp + workloads[i+1] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('broadcast'):
                    send_udp_times.append(float(row[1]))

        # with open(base_string_orca_c + workloads[i+1] + '.csv') as csvfile:
        #     results = csv.reader(csvfile, delimiter=',')
        #     for row in results:
        #         if row[0].startswith('broadcast'):
        #             orca_times_c.append(float(row[1]))

        send_udp_times_sorted = np.sort(send_udp_times) * 1000
        wait_udp_times_sorted = np.sort(wait_udp_times) * 1000
        send_orca_times_sorted = np.sort(send_orca_times) * 1000
        wait_orca_times_sorted = np.sort(wait_orca_times) * 1000
        send_orca_times_sorted_c = np.sort(send_orca_times_c) * 1000
        wait_orca_times_sorted_c = np.sort(wait_orca_times_c) * 1000

        print("\nOrca wait times 95th percentile: ", np.percentile(wait_orca_times_sorted, 95))
        print("Orca send times 95th percentile: ", np.percentile(send_orca_times_sorted, 95))
        print("UDP wait times 95th percentile: ", np.percentile(wait_udp_times_sorted, 95))
        print("UDP send times 95th percentile: ", np.percentile(send_udp_times_sorted, 95))

        print("\nOrca wait times 90th percentile: ", np.percentile(wait_orca_times_sorted, 90))
        print("Orca send times 90th percentile: ", np.percentile(send_orca_times_sorted, 90))
        print("UDP wait times 90th percentile: ", np.percentile(wait_udp_times_sorted, 90))
        print("UDP send times 90th percentile: ", np.percentile(send_udp_times_sorted, 90))

        print("\nOrca wait times 60th percentile: ", np.percentile(wait_orca_times_sorted, 60))
        print("Orca send times 60th percentile: ", np.percentile(send_orca_times_sorted, 60))
        print("UDP wait times 60th percentile: ", np.percentile(wait_udp_times_sorted, 60))
        print("UDP send times 60th percentile: ", np.percentile(send_udp_times_sorted, 60))

        send_p_udp = 1. * np.arange(len(send_udp_times)) / (len(send_udp_times) - 1)
        wait_p_udp = 1. * np.arange(len(wait_udp_times)) / (len(wait_udp_times) - 1)
        send_p_orca = 1. * np.arange(len(send_orca_times)) / (len(send_orca_times) - 1)
        wait_p_orca = 1. * np.arange(len(wait_orca_times)) / (len(wait_orca_times) - 1)
        send_p_orca_c = 1. * np.arange(len(send_orca_times_c)) / (len(send_orca_times_c) - 1)
        wait_p_orca_c = 1. * np.arange(len(wait_orca_times_c)) / (len(wait_orca_times_c) - 1)
    #ax.set_color_cycle(['red', 'green'])
        print(pos_x%2, pos_y%2)
        ax.set_ylim(top=1)
        #title = 'CDF of ' + goal.replace("_", " ") + ' with ' + str(num_receivers) + ' receivers(total ' + workload_sizes[i+1] + 'M files)'
        #ax.set_title(title)
        ax.plot(send_udp_times_sorted, send_p_udp, label='Unicast send time', linestyle='-', linewidth=DEFAULT_LINE_WIDTH, color=color_pallete[0])
        ax.plot(send_orca_times_sorted, send_p_orca, label='Orca send time',  linestyle='-', linewidth=DEFAULT_LINE_WIDTH, color=color_orca)
        ax.plot(wait_udp_times_sorted, wait_p_udp, label='Unicast wait time',  linestyle=':', linewidth=DEFAULT_LINE_WIDTH, color=color_pallete[0])
        ax.plot(wait_orca_times_sorted, wait_p_orca, label='Orca wait time',  linestyle=':', linewidth=DEFAULT_LINE_WIDTH, color=color_orca)
        #ax.plot(orca_times_sorted_c, p_orca_c, label='Orca Constrained Receiver',  linewidth=2, color=color_pallete[1])
        if mode == 'log':
            ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        #ax[pos_x%2, pos_y%2].legend(loc='best')
        ax.grid(True, which="both", ls="--", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        ax.yaxis.set_tick_params()
        ax.xaxis.set_tick_params()

        pos_y += 1
        pos_x += pos_y +1
    #for x in ax.flat:
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Fraction of Files')

    #plt.legend(loc='best')
    plt.grid(True)
    # plt.subplots_adjust(top=0.88, bottom=0.08, left=0.10, right=0.95, hspace=0.29, wspace=0.35)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    plt.tight_layout()
    ax.legend(handles, labels)
    if mode == 'log':
        savepath = '../cdf' + '_n' + str(num_receivers) +'m' + str(num_send_procs) + '_' + workloads[workload_idx] + '_log.eps'
    else: 
        savepath = '../cdf' + '_n' + str(num_receivers) +'m' + str(num_send_procs) + '_' + workloads[workload_idx] +'_.eps'
    plt.savefig(savepath, ext='eps', bbox_inches="tight")
    # plt.show(fig)

def plot_app_time_scatter(path):
    sender_runtime_udp_1 = []
    sender_runtime_udp_2 = []
    sender_runtime_udp = []
    sender_runtime_orca = []
    sender_runtime_orca_c = []
    x_axis = [int(i) for i in workload_sizes]
    for i in range (len(workloads)):
        with open(path + '/ramses/out_udp_t4_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_udp.append(float(row[1]) * 1000)
        with open(path + '/ramses/out_udp_t1_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_udp_1.append(float(row[1]) * 1000)
        with open(path + '/ramses/out_udp_t2_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_udp_2.append(float(row[1]) * 1000)
        with open(path + '/ramses/out_orca_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_orca.append(float(row[1]) * 1000)
        with open(path + '/ramses/out_orca_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_orca_c.append(float(row[1]) * 1000)
    print(sender_runtime_udp)

    a, b = best_fit(x_axis, sender_runtime_udp_1)
    fig, ax = plt.subplots(figsize=(14, 10))
    yfit = [a + b * xi for xi in x_axis]
    
    plt.plot(x_axis, sender_runtime_udp_1, '--', color='tab:blue')
    plt.scatter(x_axis, sender_runtime_udp_1, marker='s', label='Unicast: 1 core for every 4 destinations', color='tab:blue')
    #plt.plot(x_axis, yfit, '--')

    a, b = best_fit(x_axis, sender_runtime_udp_2)
    yfit = [a + b * xi for xi in x_axis]
    
    plt.plot(x_axis, sender_runtime_udp_2, '--', color='tab:green')
    plt.scatter(x_axis, sender_runtime_udp_2, marker='^', label='Unicast: 1 core for every 2 destinations', color='tab:green')
    #plt.plot(x_axis, yfit, '--')

    a, b = best_fit(x_axis, sender_runtime_udp)
    yfit = [a + b * xi for xi in x_axis]
    
    plt.plot(x_axis, sender_runtime_udp, '--', color='tab:red')
    plt.scatter(x_axis, sender_runtime_udp, marker='o', label='Unicast: 1 core for every destination', color='tab:red')
    #plt.plot(x_axis, yfit, '--')

    a, b = best_fit(x_axis, sender_runtime_orca)
    yfit = [a + b * xi for xi in x_axis]
    
    plt.plot(x_axis, sender_runtime_orca, '--', color='tab:orange')
    plt.scatter(x_axis, sender_runtime_orca, marker='*', label='Orca', color='tab:orange')
    
    #plt.plot(x_axis, yfit, '--')
    plt.title('Total send time of unicast applications with various number of CPUs per destination')
    ax.set_ylabel('Total Time (ms)')
    ax.set_xlabel('Total Data Sent (MB)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(x_axis)
    plt.savefig('../app_runtime_scatter.png')
    # plt.show(fig)

def plot_egress_data(path, mode='receivers'):
    total_udp_8 = [753.56, 1507.12, 3006.48, 6034.42, 12047.3]
    total_udp_4 = [376.78, 753.58, 1507.12, 3006.48, 6034.42]
    total_udp_12 = [1129.31, 2263.94, 4520.45, 9040.90, 18092.55]
    total_orca = [94.80, 189.62, 379.25, 758.41, 1516.83]
    x_axis_vary_receivers = [4, 8, 12]
    x_axis = [int(i) for i in workload_sizes]
    total_udp_vary_receivers = []
    total_udp_vary_receivers.append(total_udp_4[-1])
    total_udp_vary_receivers.append(total_udp_8[-1])
    total_udp_vary_receivers.append(total_udp_12[-1])
    total_orca_vary_receivers = []
    total_orca_vary_receivers.append(total_orca[-1])
    total_orca_vary_receivers.append(total_orca[-1])
    total_orca_vary_receivers.append(total_orca[-1])

    fig, ax = plt.subplots()
    # ind = np.arange(len(x_axis_vary_receivers))  # the x locations for the groups
    # width = 0.45       # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(ind, total_udp_vary_receivers, width, color=color_pallete[0])
    # #rects2 = ax.bar(ind + width, sender_runtime_udp_2, width)
    # #rects3 = ax.bar(ind + 2*width, sender_runtime_udp_4, width)
    # rects4 = ax.bar(ind + width, total_orca_vary_receivers, width, color=color_orca)
    
    # # add some text for labels, title and axes ticks
    # ax.set_ylabel('Running time (ms)')
    # ax.set_xlabel('Total file size (MB)')
    # #ax.yaxis.set_tick_params()
    # #ax.set_title('Application running time with ' + str(num_receivers) + ' receivers varying the file size loads', pad=20)
    # ax.set_xticks(ind + 0.5*width)
    # ax.set_xticklabels(workload_sizes)
    # ax.legend((rects1[0], rects4[0]), ('Unicast','Orca'))
    if mode == 'receivers':
        ax.plot(x_axis_vary_receivers, total_orca_vary_receivers, '--', marker='*', color=color_orca, linewidth=DEFAULT_LINE_WIDTH, markersize=16, label='Orca')
        ax.plot(x_axis_vary_receivers, total_udp_vary_receivers, '--', marker='o', color=color_pallete[0], linewidth=DEFAULT_LINE_WIDTH, markersize=16, label='Unicast')
        ax.set_xlabel('Number of receivers', )
        plt.xticks(x_axis_vary_receivers)
        #ax.set_yticks([1500, 6000, 12000, 18000])
    else:
        plt.plot(x_axis, total_udp_12, '--', marker='^', color=color_pallete[0], linewidth=DEFAULT_LINE_WIDTH, markersize=16, label='Unicast, 12 receivers')
        plt.plot(x_axis, total_udp_8, '--', marker='x', color=color_pallete[0], linewidth=DEFAULT_LINE_WIDTH, markersize=16, label='Unicast, 8 receivers')
        plt.plot(x_axis, total_udp_4, '--', marker='.', color=color_pallete[0], linewidth=DEFAULT_LINE_WIDTH, markersize=16, label='Unicast, 4 receivers')
        plt.plot(x_axis, total_orca, '--', marker='*', color=color_orca, linewidth=DEFAULT_LINE_WIDTH, markersize=16, label='Orca')    
        ax.set_xlabel('Workload size (MB)')
        ax.set_yticks(np.arange(max(min(total_orca), 500), max(max(total_udp_12), 20000), 2000.0))
        plt.xticks(x_axis)
        fig.autofmt_xdate()
    #ax.set_yscale('log', basey=2)
    #ax.yaxis.set_tick_params(labelsize=14)
    #ax.xaxis.set_tick_params(labelsize=14)
    ax.set_ylabel('Total transmitted traffic (MB)')
    ax.grid(True, which="both", ls="-", alpha=0.7, linewidth=1.5)
    plt.legend(loc='best')
    #plt.xticks(x_axis)
    plt.xticks(x_axis_vary_receivers)
    
    plt.savefig('../egress_data_' + mode + '.eps', ext='eps', bbox_inches="tight")
    # plt.show(fig)


def plot_scatter_workers(path):
    sender_runtime_udp = []
    sender_runtime_udp_1 = []
    sender_runtime_udp_2 = []
    sender_runtime_udp_4 = []
    sender_runtime_orca = []
    sender_runtime_orca_c = []
    x_axis = [int(i) for i in experiments_sizes]
    for i in range (len(experiments_sizes)):
        with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m1_broadcast_files_4.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_udp.append(float(row[1]) * 1000)
        if (experiments_sizes[i]==4):
            with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m1_broadcast_files_4.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_udp_4.append(float(row[1]) * 1000)
            with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m2_broadcast_files_4.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_udp_2.append(float(row[1]) * 1000)
            with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m4_broadcast_files_4.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_udp_1.append(float(row[1]) * 1000)
        elif (experiments_sizes[i]==8):
            with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m4_broadcast_files_4.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_udp_2.append(float(row[1]) * 1000)
            with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m2_broadcast_files_4.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_udp_4.append(float(row[1]) * 1000)
            # with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m8_broadcast_files_3.csv') as csvfile:
            #     results = csv.reader(csvfile, delimiter=',')
            #     for row in results:
            #         if row[0].startswith('App'):
            #             sender_runtime_udp_1.append(float(row[1]) * 1000)
        else: 
            with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m6_broadcast_files_4.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_udp_2.append(float(row[1]) * 1000)
            with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m3_broadcast_files_4.csv') as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for row in results:
                    if row[0].startswith('App'):
                        sender_runtime_udp_4.append(float(row[1]) * 1000)
            # with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m12_broadcast_files_3.csv') as csvfile:
            #         results = csv.reader(csvfile, delimiter=',')
            #         for row in results:
            #             if row[0].startswith('App'):
            #                 sender_runtime_udp_1.append(float(row[1]) * 1000)
        with open(path + '/ramses/out_orca_n' + str(experiments_sizes[i]) + '_broadcast_files_4.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_orca.append(float(row[1]) * 1000)

        # with open(path + '/ramses/out_orca_c_n' + str(experiments_sizes[i]) + '_broadcast_files_3.csv') as csvfile:
        #     results = csv.reader(csvfile, delimiter=',')
        #     for row in results:
        #         if row[0].startswith('App'):
        #             sender_runtime_orca_c.append(float(row[1]) * 1000)
    print(sender_runtime_udp)

    fig, ax = plt.subplots()
    plt.plot(x_axis, sender_runtime_udp, '-', marker='o', color=color_pallete[0], linewidth=DEFAULT_LINE_WIDTH, markersize=14, label='Unicast')
    plt.plot(x_axis, sender_runtime_udp_4, '-', marker='^', color=color_pallete[3], linewidth=DEFAULT_LINE_WIDTH, markersize=14, label='Unicast, 1 sender/4 dsts')
    plt.plot(x_axis, sender_runtime_udp_2, '-', marker='p', color=color_pallete[2], linewidth=DEFAULT_LINE_WIDTH, markersize=14, label='Unicast, 1 sender/2 dsts')
    plt.plot(x_axis[:-2], sender_runtime_udp_1, '-', marker='s', color=color_pallete[1], linewidth=DEFAULT_LINE_WIDTH, markersize=14, label='Unicast, 1 sender/dst')
    #plt.scatter(x_axis, sender_runtime_udp, marker='o', label='Unicast', color=color_pallete[0], markersize=12)
    #plt.plot(x_axis, yfit, '--')
    
    plt.plot(x_axis, sender_runtime_orca, '-', marker='*', color=color_orca, linewidth=DEFAULT_LINE_WIDTH, markersize=14, label='Orca')
    #plt.scatter(x_axis, sender_runtime_orca, marker='*', label='Orca non-contrained destination machines', color=color_orca, markersize=12)

    # plt.plot(x_axis, sender_runtime_orca_c, '--', color=color_orca_c)
    # plt.scatter(x_axis, sender_runtime_orca_c, marker='x', label='Orca constant 3 cores for destination machines', color=color_orca_c)
    #plt.plot(x_axis, yfit, '--')
    #plt.title('Total time spent for sending 704MB Wokload varying number of receivers')
    # ticks = []
    # ticks.append(min(sender_runtime_orca))
    # ticks = ticks + sender_runtime_udp_2 + sender_runtime_udp
    #ax.set_yticks(ticks)
    
    # ax.set_yticks(np.arange(0, max(max(sender_runtime_udp), 15000), 5000.0))
    ax.set_yticks(np.arange(0, 15001, 5000.0))
    
    #ax.set_yticks(np.arange(min(sender_runtime_orca), max(sender_runtime_udp), 1000.0))
    #ax.yaxis.set_tick_params(labelsize=14)
    #ax.xaxis.set_tick_params(labelsize=14)
    ax.set_ylabel('Running Time (ms)')
    ax.set_xlabel('# Receivers')
    ax.grid(True, which="both", ls="-", alpha=0.6)
    plt.legend(loc='best', fontsize=16)
    plt.xticks(x_axis)

    plt.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))

    ax.get_yaxis().get_offset_text().set_visible(False)
    ax_max = max(ax.get_yticks())
    exponent_axis = np.floor(np.log10(ax_max)).astype(int)
    ax.annotate(r'$\times$10$^{\mathregular{%i}}$'%(exponent_axis-1), xy=(.01, .98), xycoords='axes fraction', fontsize='xx-large', fontfamily='serif')

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    plt.tight_layout()

    plt.savefig('../app_runtime_scatter.eps', ext='eps', bbox_inches="tight")
    # plt.show(fig)

def plot_send_cdf_size_based(path):
    for i in range (len(workloads)):
        udp_times_small = []
        udp_times_large = []
        orca_times_small = []
        orca_times_large = []
        
        with open(path + '/ramses/out_udp_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                for idx in range(len(large_files)):
                    if row[0].find(large_files[idx]) > -1:
                        udp_times_large.append(float(row[1]))
                    elif row[0].startswith('broadcast'):
                        udp_times_small.append(float(row[1]))

        with open(path + '/ramses/out_orca_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                for idx in range(len(large_files)):
                    if row[0].find(large_files[idx]) > -1:
                        orca_times_large.append(float(row[1]))
                    elif row[0].startswith('broadcast'):
                        orca_times_small.append(float(row[1]))

        udp_times_small_sorted = np.sort(udp_times_small) * 1000
        udp_times_large_sorted = np.sort(udp_times_large) * 1000
        orca_times_small_sorted = np.sort(orca_times_small) * 1000
        orca_times_large_sorted = np.sort(orca_times_large) * 1000
        p_udp_small = 1. * np.arange(len(udp_times_small)) / (len(udp_times_small) - 1)
        p_udp_large = 1. * np.arange(len(udp_times_large)) / (len(udp_times_large) - 1)
        p_orca_small = 1. * np.arange(len(orca_times_small)) / (len(orca_times_small) - 1)
        p_orca_large = 1. * np.arange(len(orca_times_large)) / (len(orca_times_large) - 1)
        fig, ax = plt.subplots()
        #ax.set_color_cycle(['red', 'green'])
        plt.title('CDF of send times for small files (total ' + workload_sizes[i] + 'M files)')
        plt.ylim(top=1)
        #plt.xscale('log')
        plt.plot(udp_times_small_sorted, p_udp_small, label='$Unicast$')
        plt.plot(orca_times_small_sorted, p_orca_small, label='$Orca$')
        ax.set_xlabel('Send time (ms)')
        ax.set_ylabel('CDF')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('../send_times_small'+ workloads [i] +'.png')
        # plt.show(fig)

        fig, ax = plt.subplots()
        plt.title('CDF of send times large files (total ' + workload_sizes[i] + 'M files)')
        plt.ylim(top=1)
        plt.plot(udp_times_large_sorted, p_udp_large, label='$Unicast$')
        plt.plot(orca_times_large_sorted, p_orca_large, label='$Orca$')
        ax.set_xlabel('Send time (ms)')
        ax.set_ylabel('CDF')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('../send_times_large'+ workloads [i] +'.png')
        # plt.show(fig)

if __name__ == '__main__':
    path = sys.argv[1]
    print("Plotting Evaluations fom: " + path)
    for num_receivers in experiments_sizes:
        plot_app_times(path, num_receivers)
    
    # for i in range(len(workload_sizes)):
    #     plot_cdf_both(path, 12, 1, 'log', i)
    # plot_scatter_workers(path)


    # plot_egress_data(path, 'receivers')
    
    #plot_app_time_scatter(path)
    #plot_send_cdf_size_based(path)
    #plot_app_time_compare(path, 4)
    #plot_cdf(path, 12, 1, 'send_times', 'log')
    
    #plot_wait_cdf(path)
