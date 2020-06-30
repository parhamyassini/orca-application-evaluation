import os
import socket
import sys,os,time
import collections
import struct
import pickle
import csv
import sendfile
import random
import string
from multiprocessing import Process, Queue, Value, Array

RESULTS_BASE_DIR = '/local-scratch/pyassini/results/ramses/'

unix_socket_addr = '/local-scratch/pyassini/domain_socket_file'
basedir = 'broadcast_files_0'

UDP_BUF_LEN = 65507 # Max UDP message len
LOSS_TOLERANCE = 0.00
num_send_procs = 4
IP_BASE_RACK_0 = "192.168.0."
IP_BASE_RACK_1 = "192.168.1."
files = []

server_ip_net_0 = IP_BASE_RACK_0 + '1' #Ramses IP on port0
server_ip_net_1 = IP_BASE_RACK_1 + '1' #Ramses IP on port1


# Host IPs the 192.168.X.0/24 IP domain is divided to 8 subranges (max 32 containers in each subrange)
# TEST Cases

# 16 Receivers
# hosts = ["192.168.1.33", "192.168.1.34", "192.168.1.35", "192.168.1.36",\
#          "192.168.1.65", "192.168.1.66", "192.168.1.67", "192.168.1.68",\
#          "192.168.0.33", "192.168.0.34", "192.168.0.35", "192.168.0.36",\
#          "192.168.0.65", "192.168.0.66", "192.168.0.67", "192.168.0.68"]

# 12 Receivers (heterogenous)
hosts = ["192.168.1.33", "192.168.1.34", "192.168.1.35",\
         "192.168.1.65", "192.168.1.66", "192.168.1.67", "192.168.1.68",\
         "192.168.0.65", "192.168.0.66", "192.168.0.67", "192.168.0.68",\
         "192.168.0.33"]

# 8 Receivers (heterogenous)
# hosts = ["192.168.1.33", "192.168.1.34",\
#          "192.168.1.65", "192.168.1.66",\
#          "192.168.0.33",\
#          "192.168.0.65", "192.168.0.66", "192.168.0.67"]


multicast_group = ('224.1.1.1', 9001)

host_per_thread = 0
ack_times = []
total_time = 0
port = 9001
p_list = []

def send_thread(p_id, file_name, file_len, hosts, socks):
    #print ('p_id: ' + str(p_id))
    # retrans_len = int(file_len * LOSS_TOLERANCE)
    # sent_retrans_bytes = 0
    offset = p_id * host_per_thread
    with open(file_name) as reader:
        data = reader.read(UDP_BUF_LEN)
        while (data):
            for i in range(host_per_thread):
                socks[offset + i].sendto(data, (hosts[offset + i], port))
            data = reader.read(UDP_BUF_LEN)

    # while (sent_retrans_bytes < int(file_len * LOSS_TOLERANCE)):
    #     data_len = min(retrans_len - sent_retrans_bytes, UDP_BUF_LEN)
    #     retrans_data = 'z' * data_len
    #     sock.sendto(retrans_data, (host, port))
    #     sent_retrans_bytes += data_len

    #print("File Sent " + str(p_id))

def receive_thread(i, recv_sock, method, num_acks):
    ack_count = 0
    return
    if method == "udp" or method == "multicast":
        data, addr = recv_sock.recvfrom(UDP_BUF_LEN)
    elif method == "orca":
        data = recv_sock.recv(UDP_BUF_LEN)
    while data:
        print("Received : "+ str(data))
        acks_in_data = data.count("ack_" + str(i))
        #print ("ACKS IN DATA: "+ str(acks_in_data))
        ack_count += acks_in_data
        if (ack_count >= num_acks):
            print("Got all acks for file " + files[i][0])
            return
        if method == "udp" or method == "multicast":
            data, addr = recv_sock.recvfrom(UDP_BUF_LEN)
        elif method == "orca":
            data = recv_sock.recv(900)

def init(method):
    global ack_times
    global files
    global host_per_thread
    names = os.listdir(basedir)
    paths = [os.path.join(basedir, name) for name in names]
    files = [(path, os.stat(path).st_size) for path in paths]
    files.sort()
    host_per_thread = int(len(hosts)/num_send_procs)
    socks = []
    ack_times = [0] * len(files)
    tot = 0
    for file in files:
        tot += file[1]
    print ('Total bytes: ' + str(tot))
    with open(basedir + ".txt", "wb") as writer:
        pickle.dump(files, writer)
    if method == "udp":
        for host in hosts:
            sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,65536)
            socks.append(sock)
        print("UDP Send Socket Ready")
        # Listen on two network addresses for each rack
        recv_sock_net_0 = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        recv_sock_net_0.bind((server_ip_net_0,port))
        recv_sock_net_1 = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        recv_sock_net_1.bind((server_ip_net_1,port))
        print("UDP Recv Socket Ready")
        return socks, recv_sock_net_0, recv_sock_net_1
    elif method == 'multicast':
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ttl = struct.pack('b', 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        print("UDP Multicast Send Socket Ready")
        recv_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        recv_sock.bind((server_ip_net_0,port))
        print("UDP Recv Socket Ready")
        return sock, recv_sock, recv_sock
    elif method == "orca":
        sock = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            sock.connect(unix_socket_addr)
        except socket.error as err:
            print(err)
            sys.exit(1)
        print("UNIX Socket Created")
        socks.append(sock)
        return socks, sock, sock
    else:
        print("Method argument can be one of the following strings: \'orca\'  \'udp\'  \'multicast\'")
        exit(1)

def send_orca(file_name, file_len, sock):
    start_send = time.time()
    with open(file_name) as reader:
        #data = reader.read()
#        sock.sendall(data)
        data = reader.read(UDP_BUF_LEN)
        while(data):
            sock.send(data)
            data = reader.read(UDP_BUF_LEN)
      #  sent = 0
      #  while (sent < len(data)):
      #      crop_idx = min(sent+UDP_BUF_LEN, len(data))
      #      sent += sock.send(data[sent:crop_idx])
    duration_ms = 1000 * (time.time() - start_send)
    print("File Sent in : %.2f ms" %round(duration_ms, 2))

def send_multicast(file_name, sock):
    with open(file_name) as reader:
        data = reader.read()
        while (data):
            sock.sendto(data, multicast_group)
            data = reader.read(UDP_BUF_LEN)
    print("File Sent")
    
def send_files(socks, recv_sock_net_0, recv_sock_net_1, method):
    global ack_times
    ack_times = [0] * len(files)
    expected_acks_net_0 = 0
    expected_acks_net_1 = 0
    if(method == 'orca'):
        expected_acks_net_0 = len(hosts)
    else:
        expected_acks_net_0 = 5
        expected_acks_net_1 = 7
        #expected_acks_net_0 = len(hosts) / 2 # Assuming identical topology on both racks 
        #expected_acks_net_1 = len(hosts) / 2

    for j, (file_name, file_len) in enumerate(files):
        receive_proc_net_0 = Process(target=receive_thread, args=(j, recv_sock_net_0, method, expected_acks_net_0, ))
        p_list.append(receive_proc_net_0)
        receive_proc_net_0.start()
        if (method == 'udp'):
            receive_proc_net_1 = Process(target=receive_thread, args=(j, recv_sock_net_1, method, expected_acks_net_1, ))
            p_list.append(receive_proc_net_1)
            receive_proc_net_1.start()
        start_send = time.time()
        if method == 'udp':
            for i in range(num_send_procs):
                p = Process(target=send_thread, args=(i, file_name, file_len, hosts, socks, ))
                p_list.append(p)
                p.start()
        elif method == 'orca':
            p = Process(target=send_orca, args=(file_name, file_len, socks[0], ))
            p_list.append(p)
            p.start()
            #send_orca(file_name, sock)
        elif method == 'multicast':
            send_multicast(file_name, sock)
        for process in p_list:
                process.join()
        ack_times[j] = time.time() - start_send

if __name__ == '__main__':
    print("File Transfer [Sender]")
    if len(sys.argv) < 4:
        print("Run with the following arguments: <send method> <workload directory> <total run times> <num send processes>")
        exit(1)
    _method = sys.argv[1]
    basedir = sys.argv[2]
    _run_num = int(sys.argv[3])
    num_send_procs = int(sys.argv[4])
    print("Config Send Method: " + _method)
    if _method == "udp":
        result_file = RESULTS_BASE_DIR + "out_" + _method + "_n" + str(len(hosts)) + "m" + str(num_send_procs) + "_" + basedir + ".csv"
    else:
        result_file = RESULTS_BASE_DIR + "out_" + _method + "_n" + str(len(hosts)) + "_" + basedir + ".csv"
    print ("Results in: " + result_file)
    avg_total_time = 0
    os.system("taskset -p 0xff %d" % os.getpid())
    for run_idx in range(_run_num):
        total_time = 0
        start_time = time.time()
        if run_idx == 0:
            socks, recv_sock_net_0, recv_sock_net_1 = init(method=_method)
            #print (files)
            avg_ack_times = [0] * len(files)
        send_files(socks, recv_sock_net_0, recv_sock_net_1, _method)
        total_time = time.time() - start_time
        print ("Total Time: " + str(total_time))
        avg_total_time += total_time / _run_num
        for j in range(len(files)):
            avg_ack_times[j] += ack_times[j] / _run_num
        time.sleep(7) # Sleep between running each experiment
    with open(result_file, 'wb') as file:
        fieldnames = ['name', 'time']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i, time in enumerate(ack_times):
            writer.writerow({
                'name': files[i][0],
                'time': avg_ack_times[i]
            })
        writer.writerow({
                    'name': 'App Runing Time',
                    'time': avg_total_time
                })
sys.exit()
