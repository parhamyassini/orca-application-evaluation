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
import matplotlib as mpl
import math
import pandas as pd
import seaborn as sns

# Line Styles
DEFAULT_LINE_WIDTH = 4
ALTERNATIVE_LINE_WIDTH = 5
MEDIUM_LINE_WIDTH = 3
SMALL_LINE_WIDTH = 2
LINE_STYLES = ['-', '--', '-.', ':']

# Font
TEX_ENABLED = False
TICK_FONT_SIZE = 24
AXIS_FONT_SIZE = 24
LEGEND_FONT_SIZE = 22
CAP_SIZE = LEGEND_FONT_SIZE / 2

FONT_DICT = {'family': 'serif', 'serif': 'Times New Roman'}

DEFAULT_RC = {'lines.linewidth': DEFAULT_LINE_WIDTH,
              'axes.labelsize': AXIS_FONT_SIZE,
              'xtick.labelsize': TICK_FONT_SIZE,
              'ytick.labelsize': TICK_FONT_SIZE,
              'legend.fontsize': LEGEND_FONT_SIZE,
              'text.usetex': TEX_ENABLED,
              # 'ps.useafm': True,
              # 'ps.use14corefonts': True,
              'font.family': 'sans-serif',
              # 'font.serif': ['Helvetica'],  # use latex default serif font
              }

SCATTER_MARKER_DIAMETER = 64
# Controller Timeout
k_arr = [1, 5, 10, 15, 20, 25, 30]

size_arr = [64, 128, 256, 512, 1024]
cores_arr = [1, 2, 3, 4, 5, 6, 7, 8]

color_pallete = ['#e69d00', '#0071b2', '#009e74', '#cc79a7', '#d54300', '#994F00', '#000000']
marker_list = ['o', '*', 's', '^', 'D']
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

sns.set_context(context='paper', rc=DEFAULT_RC)
sns.set_style(style='ticks')
plt.rc('text', usetex=TEX_ENABLED)
plt.rc('ps', **{'fonttype': 42})
plt.rc('legend', handlelength=1., handletextpad=0.1)
fig, ax = plt.subplots()

def plot_throughput_normal(path):
    mean_list = []
    err_list = []
    sns.set_context(context='paper', rc=DEFAULT_RC)
    #sns.set_style(style='ticks')
    plt.rc('text', usetex=TEX_ENABLED)
    plt.rc('ps', **{'fonttype': 42})
    #plt.rc('legend', handlelength=1., handletextpad=0.1)
    fig, ax = plt.subplots()
    for plot_idx, size in enumerate(size_arr):
        results_bits = []
        results_times = []
        file_name = path + '/57/throughput_size_' + str(size) + '_k_1.csv'
        with open(file_name) as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(results):
                if idx == 0:
                    continue
                results_bits.append(float(row[2]))
                results_times.append(int(row[0]))
                # if idx == 200000:
                #     continue
        throughput_list = []
        ns_count = 0
        second_count = 0.0
        bits_count = 0
        for i in range(1, len(results_times)):
            diff_ns = results_times[i] - results_times[i-1]
            ns_count += diff_ns
            bits_count += results_bits[i]
            if (ns_count >= 1000000000):
                throughput = float(bits_count) / ns_count
                throughput_list.append(throughput)
                bits_count = 0
                ns_count = 0
        mean = np.mean(throughput_list)
        std = np.std(throughput_list)
        mean_list.append(mean)
        err_list.append(std)
    print(mean_list)
    print(err_list)
    x_pos = np.arange(len(size_arr))
    ax.set_xticks(x_pos)
    ax.set_ylabel('Receive throughput (Gbps)')
    ax.set_xlabel('Packet size (B)')
    ax.set_xticklabels([str(size) for size in size_arr])
    ax.bar(x_pos, mean_list, yerr=err_list, align='center', ecolor='black', capsize=TICK_FONT_SIZE, zorder=3, color=color_pallete[0])
    ax.grid(zorder=0)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    plt.savefig('../throughput_normal.eps', ext='eps', bbox_inches="tight")
    plt.show()


def plot_loss_percentage(path):
    sns.set_context(context='paper', rc=DEFAULT_RC)
    plt.rc('text', usetex=TEX_ENABLED)
    plt.rc('ps', **{'fonttype': 42})
    mpl.rcParams["font.size"] = LEGEND_FONT_SIZE
    #fig, ax = plt.subplots(figsize=(6.5, 4))
    
    for size_index, size in enumerate(size_arr):
        loss_rate_arr = []
        for plot_idx, k in enumerate(k_arr):
            min_throughput = 10
            min_throughput_index = 0
            failure_point_ns = 0.0
            failure_point_idx = 0
            recovery_point_idx = 0
            results_bits = []
            results_times = []
            file_name = path + '/57/throughput_size_' + str(size) + '_k_' + str(k) + '.csv'
            with open(file_name) as csvfile:
                results = csv.reader(csvfile, delimiter=',')
                for idx, row in enumerate(results):
                    if idx == 0:
                        continue
                    if k==30 and size==128 and idx < 14000: #Workaround for long experiment duration
                        continue
                    results_bits.append(float(row[2]))
                    results_times.append(int(row[0]))
            #print(results_times)
            # print(results_bits[0])
            ns_count = 0
            second_count = 0.0
            bits_count = 0
            x_axis = []
            y_axis = []
            for i in range(1, len(results_times)):
                diff_ns = results_times[i] - results_times[i-1]
                ns_count += diff_ns
                bits_count += results_bits[i]
                # Find the failure point
                if results_bits[i] == 0 and failure_point_ns == 0.0:
                    failure_point_ns = second_count + (ns_count / 1000000000)
                    failure_point_idx = i

                if failure_point_idx != 0 and recovery_point_idx == 0:
                    if (results_bits[i] > 0.0):
                        recovery_point_idx = i
                        # print(recovery_point_idx)
                if (ns_count >= 1000000000):
                    second = second_count + 1
                    x_axis.append(second)
                    throughput = float(bits_count) / ns_count
                    y_axis.append(throughput)
                    if throughput < min_throughput:
                        min_throughput = throughput
                        min_throughput_index = len(x_axis) - 1
                        #print ("min index: " +str(min_throughput_index))
                    second_count += float(ns_count)/1000000000
                    bits_count = 0
                    ns_count = 0
                    #print(second_count)
            #print(plot_idx)
            loss_percentage = 100*(float((y_axis[min_throughput_index - 1] - y_axis[min_throughput_index]))/y_axis[min_throughput_index - 1])
            loss_rate_arr.append(loss_percentage)
        plt.scatter(k_arr, loss_rate_arr,  color=color_pallete[size_index], marker=marker_list[size_index], s=64, label='Packet size = ' + str(size_arr[size_index]))
        print (loss_rate_arr)
        if size_index == len(size_arr) -1 :
            #plt.title('Loss rate for differnet packet sizes')
            #ax.set_xlabel('T (ms)')
            ax.set_ylabel('Loss percentage (%)')
            #ax.set_yticks(np.arange(0, 12, 1))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #plt.legend(loc='best')
            ax.set_yticks(loss_rate_arr)
            ax.set_xticks(k_arr)
            # labels = [item.get_text() for item in ax.get_xticklabels()]

            # empty_string_labels = ['']*len(labels)
            # ax.set_xticklabels(empty_string_labels)

            ax.grid(True, which="both", ls="--", alpha=0.6)
            plt.legend(loc='best')
            plt.savefig('../loss_rates.eps', ext='eps', bbox_inches="tight")
            plt.show(fig)

def plot_throughput_failure(path, size=1024, metric='bps', size_index=0, k=1):
    all_pkts_arr = []
    all_bits_arr = []
    all_timestamp_arr = []
    sns.set_context(context='paper', rc=DEFAULT_RC)
    #sns.set_style(style='ticks')
    plt.rc('text', usetex=TEX_ENABLED)
    plt.rc('ps', **{'fonttype': 42})
    #plt.rc('legend', handlelength=1., handletextpad=0.1)
    mpl.rcParams["font.size"] = LEGEND_FONT_SIZE
    #fig, ax = plt.subplots()
    

    min_throughput = 10
    min_throughput_index = 0
    failure_point_ns = 0.0
    failure_point_idx = 0
    recovery_point_idx = 0
    results_bits = []
    results_times = []
    file_name = path + '/57/throughput_size_' + str(size) + '_k_' + str(k) + '.csv'
    with open(file_name) as csvfile:
        results = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(results):
            if idx == 0:
                continue
            if k==30 and size==128 and idx < 14000: #Workaround for long experiment
                continue
            results_bits.append(float(row[2]))
            results_times.append(int(row[0]))
    #print(results_times)
    print(results_bits[0])
    ns_count = 0
    second_count = 0.0
    bits_count = 0
    x_axis = []
    y_axis = []
    for i in range(1, len(results_times)):
        diff_ns = results_times[i] - results_times[i-1]
        ns_count += diff_ns
        bits_count += results_bits[i]
        # Find the failure point
        if results_bits[i] == 0 and failure_point_ns == 0.0:
            failure_point_ns = second_count + (ns_count / 1000000000)
            failure_point_idx = i

        if failure_point_idx != 0 and recovery_point_idx == 0:
            if (results_bits[i] > 0.0):
                recovery_point_idx = i
                print(recovery_point_idx)
        if (ns_count >= 1000000000):
            second = second_count + 1
            x_axis.append(second)
            throughput = float(bits_count) / ns_count
            y_axis.append(throughput)
            if throughput < min_throughput:
                min_throughput = throughput
                min_throughput_index = len(x_axis) - 1
                #print ("min index: " +str(min_throughput_index))
            second_count += float(ns_count)/1000000000
            bits_count = 0
            ns_count = 0
            #print(second_count)
    #print(plot_idx)
    # print('Failure time: %.2f' % failure_point_ns)
    # print('Min throughput: ' + str(y_axis[min_throughput_index]))
    # print('Normal throughput: ' + str(y_axis[min_throughput_index + 1]))
    # print('Controller response time (ns): ' + str(results_times[recovery_point_idx] - results_times[failure_point_idx]))
    
    centered_x = [int(item - (min_throughput_index - 2)) for item in x_axis[min_throughput_index-2:min_throughput_index+3]]
    centered_y = y_axis[min_throughput_index-2:min_throughput_index+3]
    plt.plot(centered_x, centered_y, '-', color=color_pallete[size_index], label=str(size) +' B Packets' )
    print (centered_y)
    print(centered_x)
    ax.annotate('Throughput drop: %.3f Gbps' % (y_axis[min_throughput_index - 2] - min_throughput),
        xy=(min_throughput_index - (min_throughput_index - 3), min_throughput), 
        xytext=(min_throughput_index - (min_throughput_index - 3), min_throughput-0.5),
        horizontalalignment="center", 
        arrowprops=dict(arrowstyle='<|-', color='black', lw=DEFAULT_LINE_WIDTH))
    # #plt.plot(x_axis, ideal, '--', color='tab:red', label='Line rate (Mbps)')
        #x_axis.append(failure_point_ns)
    #     plt.scatter(failure_point_ns, 0, marker='x', label='Failure point', color='tab:blue')
    # plt.plot(x_axis, yfit, '--')
    
    #plt.plot(x_axis, yfit, '--')
    # plt.title('Receiver throughput during agent failure, packet size = ' + str(size))
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Throughput (Gbps)')

    ax.set_yticks(np.arange(8, 11, 1))
    ax.set_xticks(centered_x)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc='lower right')
    ax.grid(True, which="both", ls="--", alpha=0.6)
    #plt.grid(True)
    ticks = []

    #plt.xticks([int(i/10) for i in range(0, len(x_axis), 10)])
    plt.savefig('../throughput_size_' + str(size) + '_best.eps', ext='eps', bbox_inches="tight")
    plt.show(fig)

def plot_cpu(path):
    cpu_cyle_arr = []
    for size in size_arr:
        file_name = path + '/56/cpu_size_' + str(size) + '.csv'
        df = pd.read_csv(file_name)
        mean_cycle_per_pkt = df[df.cycle_per_pkt != 0].cycle_per_pkt.mean()
        cpu_cyle_arr.append(mean_cycle_per_pkt)
    print(cpu_cyle_arr)
    x_axis = size_arr
    fig, ax = plt.subplots(figsize=(14, 10))
    
    plt.scatter(x_axis, cpu_cyle_arr, marker='s', s=100, color=color_pallete[0])
    
    plt.title('Orca agent processing overhead for different packet sizes')
    ax.set_xlabel('Packet size (Byte)')
    ax.set_ylabel('Processing overhead (CPU cycles per packet)')
    #ax.set_yticks(np.arange(0, 11000, 1000))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc='lower right')
    ax.grid(True, which="both", ls="--", alpha=0.6)
    #plt.grid(True)
    plt.xticks(x_axis)
    #plt.xscale('log', basex=2)
    
    plt.savefig('../processing_overhead.png')
    plt.show(fig)

def plot_cdf_latency(path, exp='10g'):
    data_orca = []
    data_normal = []
    
    if (exp == '10g'):
        res_filename_orca_base = 'latency_orca_10g_'
        res_filename_normal_base = 'latency_normal_10g_'
    else:
        res_filename_orca_base = 'latency_orca_inc_'
        res_filename_normal_base = 'latency_normal_inc_'
    for size in size_arr:
        res_orca = []
        res_normal = []
        f = open(path + res_filename_orca_base + str(size) + '_probe_0.txt', "r")
        for i,latency in enumerate(f):
            if i==0:
                print(latency)
            else:
                if i==99:
                    print("99th : " +str(latency))
                res_orca.append(float(latency))
        f = open(path + res_filename_normal_base + str(size) + '_probe_0.txt', "r")
        for i,latency in enumerate(f):
            if i==0:
                print(latency)
            else:
                if i==99:
                    print("99th : " +str(latency))
                res_normal.append(float(latency))
        data_orca.append(res_orca)
        data_normal.append(res_normal)
    
    ticks = ['64', '128', '256', '512', '1024']

    def set_box_color(bp, color, hatch=None):
        plt.setp(bp['boxes'], linewidth=1)
        plt.setp(bp['whiskers'], linewidth=1)
        plt.setp(bp['caps'], linewidth=1)
        plt.setp(bp['medians'], color='black', linewidth=1)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            if hatch:
                patch.set_hatch(hatch)
        #plt.setp(bp['fliers'], color=color, marker='+')
    
    sns.set_context(context='paper', rc=DEFAULT_RC)
    sns.set_style(style='ticks')
    plt.rc('font', **FONT_DICT)
    plt.rc('ps', **{'fonttype': 42})
    plt.rc('pdf', **{'fonttype': 42})
    plt.rc('mathtext', **{'fontset': 'cm'})
    plt.rc('ps', **{'fonttype': 42})
    plt.rc('legend', handlelength=1., handletextpad=0.1)
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
    #plt.rcParams['text.usetex'] = True
    # fig, ax = plt.subplots(figsize=(9, 5))
    fig, ax = plt.subplots()
    bpl = plt.boxplot(data_orca, patch_artist=True, positions=np.array(range(len(data_orca)))*2.0-0.3, sym='')
    bpr = plt.boxplot(data_normal, patch_artist=True, positions=np.array(range(len(data_normal)))*2.0+0.3, sym='')
    set_box_color(bpl, color_pallete[0], hatch='xx') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, color_pallete[1], hatch='//')

    # draw temporary red and blue lines and use them to create a legend
    # plt.plot([], c=color_pallete[0], label='Orca')
    # plt.plot([], c=color_pallete[1], label='Network-based Multicast')
    # plt.legend()
    ax.legend((bpl['boxes'][0], bpr['boxes'][0]), ('Orca', 'Network-based Multicast'))

    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.yticks(np.arange(0, 30, 5))
    plt.xlim(-1, len(ticks)*2-1)
    #plt.ylim(0, 8)
    # plt.xlabel('Packet size (Bytes)')
    # plt.ylabel('Packet receive latency (\u03BCs)')
    ax.set_xlabel('Packet Size (Bytes)')
    ax.set_ylabel('Packet Latency (%ss)' % r'$\mu$')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig('../latency_' + str(exp) + '.eps', ext='eps', bbox_inches="tight")
    # plt.show()


def plot_agent_load():
    x_axis = cores_arr
    # Manually extracted from results_load_test !
    throughput_1024 = ([40.0218940] * len(cores_arr))
    
    throughput_512 = [37.3714957659] + [40.010905286] * (len(cores_arr) -1)
    
    throughput_256 = [19.7648949365, 39.3145943708] + [40.0220538427] * (len(cores_arr) - 2)
    throughput_128 = [10.5430351136, 21.1075725034, 30.3853887471] + [40.0188566] * (len(cores_arr) - 3)
    throughput_64 = [6.21866155276, 12.0620738732, 17.1684742196, 22.3771506654, 28.7451668127, 31.0031466775, 35.0225459591, 40.0166995541]
    
    sns.set_context(context='paper', rc=DEFAULT_RC)
    # sns.set_style(style='ticks')
    plt.rc('font', **FONT_DICT)
    plt.rc('ps', **{'fonttype': 42})
    plt.rc('pdf', **{'fonttype': 42})
    plt.rc('mathtext', **{'fontset': 'cm'})
    plt.rc('ps', **{'fonttype': 42})
    # plt.rc('legend', handlelength=1., handletextpad=0.1)

    # sns.set_context(context='paper', rc=DEFAULT_RC)
    #sns.set_style(style='ticks')
    # plt.rc('text', usetex=TEX_ENABLED)
    # plt.rc('ps', **{'fonttype': 42})
    #plt.rc('legend', handlelength=1., handletextpad=0.1)
    fig, ax = plt.subplots()
    plt.plot(x_axis, throughput_1024, label='Pkt Size=1024B', marker='.', markersize=TICK_FONT_SIZE)
    plt.plot(x_axis, throughput_512, label='Pkt Size=512B', marker='.', markersize=TICK_FONT_SIZE)
    plt.plot(x_axis, throughput_256, label='Pkt Size=256B', marker='.', markersize=TICK_FONT_SIZE)
    plt.plot(x_axis, throughput_128, label='Pkt Size=128B', marker='.', markersize=TICK_FONT_SIZE)
    plt.plot(x_axis, throughput_64, label='Pkt Size=64B', marker='.', markersize=TICK_FONT_SIZE)
    ax.set_xlabel('# CPU Cores')
    ax.set_ylabel('Throughput (Gbps)')

    ax.set_yticks(np.arange(0, 41, 5))
    ax.set_xticks(x_axis)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('../agent_load.eps', ext='eps', bbox_inches="tight")
    # plt.show()

def plot_failover_delay(mode='bar'):
    # Extracted from google sheet calculations
    y_axis = [1.04, 5.07, 9.28, 15.06, 20.75,24.75, 30.14]
    y_std = [0.406853887, 0.8856466732, 0.7557480249, 0.2358643558, 0.8181474411 ,0.3507938208 ,0.629904846]
    x_axis = k_arr

    # sns.set_style(style='ticks')
    plt.rc('font', **FONT_DICT)
    plt.rc('ps', **{'fonttype': 42})
    plt.rc('pdf', **{'fonttype': 42})
    plt.rc('mathtext', **{'fontset': 'cm'})
    plt.rc('ps', **{'fonttype': 42})
    # plt.rc('legend', handlelength=1., handletextpad=0.1)

    # sns.set_context(context='paper', rc=DEFAULT_RC)
    #sns.set_style(style='ticks')
    # plt.rc('text', usetex=TEX_ENABLED)
    # plt.rc('ps', **{'fonttype': 42})
    #plt.rc('legend', handlelength=1., handletextpad=0.1)
    
    fig, ax = plt.subplots()
    if mode == 'bar':
        ax.bar(x_axis, y_axis, ecolor='black', capsize=CAP_SIZE, color=color_pallete[1], yerr=y_std, width = MEDIUM_LINE_WIDTH, zorder=3)
    elif mode == 'line':
        ax.plot(x_axis, y_axis, color=color_pallete[1], marker='o', markersize=LEGEND_FONT_SIZE, zorder=2)
        ax.plot(x_axis, x_axis, '--', color='#FF0000', linewidth=DEFAULT_LINE_WIDTH, zorder=3)
    ax.set_xlabel('Heartbeat Timeout Interval (ms)')
    ax.set_ylabel('Flow Disruption Duration (ms)')
    ax.set_yticks([int(i) for i in y_axis])
    ax.set_xticks(x_axis)
    ax.grid(True, which="both", ls="--", alpha=0.6, zorder=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if mode =='line':
        ax.set_ylim(0, int(max(y_axis)) + 1)
        ax.set_xlim(0, int(max(y_axis)) + 1)
    plt.tight_layout()
    plt.savefig('../failover_delay_' + mode + '.eps', ext='eps', bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    path = sys.argv[1]
    print("Plotting Evaluations fom: " + path)
    # Paper
    # plot_loss_percentage(path)
    # Paper
    plot_failover_delay(mode='line')
    # Paper
    # plot_throughput_normal(path)
    # Paper
    # plot_agent_load()

    # latency results in a different path
    # Paper
    #plot_cdf_latency(path, 'inc')
    
    #plot_throughput_failure(path, size=size_arr[0], size_index=0, k=1) # 64B k=1ms
    # Paper
    # plot_throughput_failure(path, size=size_arr[4], size_index=1, k=1) # 1024B k=1ms
    # Paper
    # plot_cpu(path)
