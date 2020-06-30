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
import math

workloads = ['broadcast_files_0', 'broadcast_files_1', 'broadcast_files_2', 'broadcast_files_3', 'broadcast_files_4']
workload_sizes = ['88', '176', '352', '704', '1408']
udp_experiments = ['n4m1', 'n8m1', 'n12m1']
experiments_sizes = [4, 8, 12]
large_files = ['broadcast_12', 'broadcast_15', 'broadcast_18', 'broadcast_21', 'broadcast_24', 'broadcast_27', 'broadcast_30', 'broadcast_33', 'broadcast_6', 'broadcast_9']
color_pallete = ['tab:blue', 'tab:green', 'tab:red', 'tab:grey']
color_orca = 'tab:orange'
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

def plot_app_time(path):
    sender_runtime_udp = []
    sender_runtime_orca = []
    for i in range (len(workloads)):
        with open(path + '/ramses/out_udp_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_udp.append(float(row[1]) * 1000)
        with open(path + '/ramses/out_orca_' + workloads[i] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_orca.append(float(row[1]) * 1000)
    print(sender_runtime_udp)
    print(sender_runtime_orca)
    ind = np.arange(len(sender_runtime_orca))  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, sender_runtime_udp, width, color='r')
    rects2 = ax.bar(ind + width, sender_runtime_orca, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Running Time (ms)')
    ax.set_xlabel('Total File Size (MB)')
    ax.set_title('Application running time varying the file size loads', pad=20)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(workload_sizes)
    ax.legend((rects1[0], rects2[0]), ('Unicast', 'Orca'))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.savefig('../app_runtime.png')
    plt.show()

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
    plt.show()

def plot_app_times(path, num_receivers):
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
    width = 0.2       # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(ind, sender_runtime_udp_1, width)
    #rects2 = ax.bar(ind + width, sender_runtime_udp_2, width)
    #rects3 = ax.bar(ind + 2*width, sender_runtime_udp_4, width)
    rects4 = ax.bar(ind + width, sender_runtime_orca, width)
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Running Time (ms)')
    ax.set_xlabel('Total File Size (MB)')
    ax.set_title('Application running time with ' + str(num_receivers) + ' receivers varying the file size loads', pad=20)
    ax.set_xticks(ind + 0.5*width)
    ax.set_xticklabels(workload_sizes)
    ax.legend((rects1[0], rects4[0]), 
        ('Unicast',
         'Orca'))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    autolabel(rects4)
    plt.savefig('../app_runtime_n' + str(num_receivers) + 'm1.png')
    plt.show()

def plot_cdf(path, num_receivers, num_send_procs, goal="send_times", mode="linear"):
    file_name_udp = 'out_udp_n'+ str(num_receivers) +'m' + str(num_send_procs) + '_'
    file_name_orca = 'out_orca_n' + str(num_receivers) + '_'
    file_name_orca_c = 'out_orca_c_n' + str(num_receivers) + '_'
    base_string_udp = path + '/ramses/' + file_name_udp
    base_string_orca = path + '/ramses/' + file_name_orca
    base_string_orca_c = path + '/ramses/' + file_name_orca_c
    if goal == "wait_times":
        base_string_udp = path + '/56/container_1/' + file_name_udp
        base_string_orca = path + '/56/container_1/' + file_name_orca
        base_string_orca_c = path + '/56/container_1/' + file_name_orca_c
    fig, ax = plt.subplots(2,2, figsize=(13, 9.5))
    pos_x = 0
    pos_y = 0
    for i in range (len(workloads)-1):
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
        with open(base_string_orca_c + workloads[i+1] + '.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('broadcast'):
                    orca_times_c.append(float(row[1]))

        udp_times_sorted = np.sort(udp_times) * 1000
        orca_times_sorted = np.sort(orca_times) * 1000
        orca_times_sorted_c = np.sort(orca_times_c) * 1000
        p_udp = 1. * np.arange(len(udp_times)) / (len(udp_times) - 1)
        p_orca = 1. * np.arange(len(orca_times)) / (len(orca_times) - 1)
        p_orca_c = 1. * np.arange(len(orca_times_c)) / (len(orca_times_c) - 1)
    #ax.set_color_cycle(['red', 'green'])
        print(pos_x%2, pos_y%2)
        ax[pos_x%2, pos_y%2].set_ylim(top=1)
        title = 'CDF of ' + goal.replace("_", " ") + ' with ' + str(num_receivers) + ' receivers(total ' + workload_sizes[i+1] + 'M files)'
        ax[pos_x%2, pos_y%2].set_title(title)
        ax[pos_x%2, pos_y%2].plot(udp_times_sorted, p_udp, label='$Unicast$')
        ax[pos_x%2, pos_y%2].plot(orca_times_sorted, p_orca, label='$Orca$')
        ax[pos_x%2, pos_y%2].plot(orca_times_sorted_c, p_orca_c, label='Orca Constrained')
        if mode == 'log':
            ax[pos_x%2, pos_y%2].set_xscale('log')
        #ax[pos_x%2, pos_y%2].legend(loc='best')
        ax[pos_x%2, pos_y%2].grid(True)
        handles, labels = ax[pos_x%2, pos_y%2].get_legend_handles_labels()
        
        pos_y += 1
        pos_x += pos_y +1
    for x in ax.flat:
        x.set_xlabel(goal.replace("_", " ") + ' (ms)')
        x.set_ylabel('CDF')

    #plt.legend(loc='best')
    plt.grid(True)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.29,
                    wspace=0.35)
    fig.legend(handles, labels, loc='upper center', ncol=3)
    if mode == 'log':
        savepath = '../' + goal + '_c_n' + str(num_receivers) +'m' + str(num_send_procs) + '_log.png'
    else: 
        savepath = '../' + goal + '_c_n' + str(num_receivers) +'m' + str(num_send_procs) +'_.png'
    plt.savefig(savepath)
    plt.show(fig)

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
    plt.show(fig)

def plot_scatter_workers(path):
    sender_runtime_udp = []
    sender_runtime_orca = []
    sender_runtime_orca_c = []
    x_axis = [int(i) for i in experiments_sizes]
    for i in range (len(experiments_sizes)):
        with open(path + '/ramses/out_udp_n' + str(experiments_sizes[i]) +'m1_broadcast_files_3.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_udp.append(float(row[1]) * 1000)
        with open(path + '/ramses/out_orca_n' + str(experiments_sizes[i]) + '_broadcast_files_3.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_orca.append(float(row[1]) * 1000)
        with open(path + '/ramses/out_orca_c_n' + str(experiments_sizes[i]) + '_broadcast_files_3.csv') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for row in results:
                if row[0].startswith('App'):
                    sender_runtime_orca_c.append(float(row[1]) * 1000)
    print(sender_runtime_udp)

    fig, ax = plt.subplots(figsize=(14, 10))
    plt.plot(x_axis, sender_runtime_udp, '--', color='tab:blue')
    plt.scatter(x_axis, sender_runtime_udp, marker='o', label='Unicast', color='tab:blue')
    #plt.plot(x_axis, yfit, '--')
    
    plt.plot(x_axis, sender_runtime_orca, '--', color='tab:orange')
    plt.scatter(x_axis, sender_runtime_orca, marker='*', label='Orca non-contrained destination machines', color='tab:orange')

    plt.plot(x_axis, sender_runtime_orca_c, '--', color=color_orca_c)
    plt.scatter(x_axis, sender_runtime_orca_c, marker='x', label='Orca constant 3 cores for destination machines', color=color_orca_c)
    #plt.plot(x_axis, yfit, '--')
    plt.title('Total time spent for sending 704MB Wokload varying number of receivers')
    ax.set_ylabel('Total Time (ms)')
    ax.set_xlabel('Number of receivers')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(x_axis)
    plt.savefig('../app_runtime_c_scatter.png')
    plt.show(fig)

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
        plt.show(fig)

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
        plt.show(fig)

if __name__ == '__main__':
    path = sys.argv[1]
    print("Plotting Evaluations fom: " + path)
    plot_scatter_workers(path)
    #plot_app_time_scatter(path)
    #plot_send_cdf_size_based(path)
    #plot_app_time_compare(path, 8)
    #plot_app_times(path, 12)
    #plot_cdf(path, 8, 1, 'wait_times', 'linear')
    #plot_wait_cdf(path)

