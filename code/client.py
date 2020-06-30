import socket
import sys,os,time
import struct
import csv
import pickle
import shutil
from multiprocessing import Process, Queue, Value, Array, Event

RESULTS_BASE_DIR = '/var/run/sockets/results/'

server_ip = "" #ramses IP
host_ip = "" #Receiver host IP
multicast_group = '224.1.1.1'
server_address = ('', 9001)
NUM_RECEIVER_THREADS = 1
ACCEPTABLE_PORTION = 1
UDP_BUFFER_LEN = 65507
UNIX_SOCK_BUFFER_LEN = 16 * 800
UNIX_SOCK_BASE_DIR = '/var/run/sockets/'

port = 9001
unix_socket_addr = '../domain_socket_file'
files = []
total_written_bytes = []
total_time = 0

def init(method):
    global files
    global total_written_bytes
    global server_ip
    with open(broadcast_name, "rb") as reader:
        files = pickle.load(reader)
    try:
        unix_socket_addr = UNIX_SOCK_BASE_DIR + os.getenv('CONTAINER_NAME', None)
    except:
        print("Warning: No container name set using Unix socket address for physical host")
        unix_socket_addr = '../domain_socket_file'
    server_ip = host_ip[:10] + '1' #Ramses has 192.168.1.1 or 192.168.0.1 depending on Rack subnet
    #print(host_ip)
    #print (server_ip)
    sock_array = []
    print("Config Host IP: " + host_ip)
    if method == "udp":
        sock_send = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("UDP Send Socket Ready")
        for i in range(NUM_RECEIVER_THREADS):
            sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host_ip,port))
            sock_array.append(sock)
        print("UDP Receive Socket Ready")
        return sock_array, sock_send
    elif method == "multicast":
        sock_send = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("UDP Send Socket Ready")
        sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.bind(server_address)
        group = socket.inet_aton(multicast_group)
        mreq = struct.pack('4sL', group, socket.INADDR_ANY)
        sock.setsockopt(
            socket.IPPROTO_IP,
            socket.IP_ADD_MEMBERSHIP,
            mreq)
        # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # try:
        #     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # except AttributeError:
        #     pass
        # sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_TTL, 20)
        # sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_LOOP, 1)
        # sock.bind(server_address)
        # intf = socket.gethostbyname(socket.gethostname())
        # sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(intf))
        # sock.setsockopt(socket.SOL_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(multicast_group) + socket.inet_aton(intf))
        print("UDP Multicast Socket Receive Ready")
        return sock, sock_send
    elif method == "orca":
        sock = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            sock.connect(unix_socket_addr)
        except socket.error as err:
            print(err)
            sys.exit(1)
        print("UNIX Socket Connected")
        sock_array.append(sock)
        return sock_array, sock
    else:
        print("Method argument can be one of the following strings: \'orca\'  \'udp\'  \'multicast\'")
        exit (1)

def write_files(rec_buffer, sock, sock_send, method, end_time, finish_times):
    total_bytes = 0
    file_idx = 0
    writer = open(files[file_idx][0],"wb")
    chunks = []
    total_rec = 0
    processed_bytes = 0
    buffer_index = 0

    while file_idx < len(files):
        processed_bytes = 0
        try:
            data = rec_buffer.get()
        except IndexError as e:
            continue
        
        received_len = len(data)
        buffer_index += 1
        while processed_bytes < received_len:
            crop_index = min(len(data), files[file_idx][1] - total_written_bytes[file_idx])
            processed_bytes += crop_index
            #writer.write(data[:crop_index])
            #if (data[:crop_index].find(file_contents[file_idx]) == -1):
            #    print(data[:crop_index])
            #if (data[:crop_index].find(file_contents[file_idx]) != -1):
            chunks.append(data[:crop_index])
            total_written_bytes[file_idx] += len(data[:crop_index])
            data = data[crop_index:]

            if total_written_bytes[file_idx] >= ACCEPTABLE_PORTION * files[file_idx][1]: #FINISHED RECEIVING THIS FILE
                writer.write(b''.join(chunks))
                finish_times[file_idx] = time.time()
                print("Received File: {}".format(files[file_idx][0]))
                if method == "udp" or method == "multicast":
                    sock_send.sendto("ack_" + str(file_idx), (server_ip,port))
                elif method == "orca":
                    sock.sendall("ack_" + str(file_idx))
                if file_idx < len(files)-1:
                    file_idx += 1
                    writer = open(files[file_idx][0],"wb")
                    chunks = []
                else:
                    end_time.value = time.time()
                    return

def count_bytes(file_received_events, sock, sock_send, method, end_time, finish_times):
    total_bytes = 0
    file_idx = 0
    chunks = []
    last_file_byte_count = 0
    while file_idx < len(files):
        file_received_events[file_idx].wait()
        finish_times[file_idx] = time.time()
        print("Received File: {}".format(files[file_idx][0]))
        if method == "udp" or method == "multicast":
            sock_send.sendto("ack_" + str(file_idx), (server_ip,port))
        elif method == "orca":
            sock_send.sendall("ack_" + str(file_idx))
        if file_idx < len(files)-1:
            file_idx += 1
        else:
            end_time.value = time.time()
            return

def receive_thread(file_received_events, sock, method, start_time):
    #chunks = []
    #bytes_received = 0
    try:
        file_idx = 0
        total_bytes_received = 0
        last_file_byte_count = 0
        #chunk_num = int(FILE_SIZE / CHUNK_SIZE)
        if method == "udp" or method == "multicast":
            data = sock.recv(UDP_BUFFER_LEN)
        elif method == "orca":
            data = sock.recv(UNIX_SOCK_BUFFER_LEN)
        if start_time.value == 0: #only the first thread to reach here starts the timer
            start_time.value = time.time()
        while file_idx < len(files):
            total_bytes_received += len(data)
            #print(total_bytes_received)
            if total_bytes_received - last_file_byte_count >= files[file_idx][1]:
                #print(total_bytes_received)
                #print(total_bytes_received)
                #print(total_bytes_received)
                file_received_events[file_idx].set()
                last_file_byte_count += files[file_idx][1]
                if file_idx < len(files)-1:
                    file_idx += 1
            if method == "udp" or method == "multicast":
                data = sock.recv(UDP_BUFFER_LEN)
            elif method == "orca":
                data = sock.recv(UNIX_SOCK_BUFFER_LEN)
    except KeyboardInterrupt:
        print('TOT REC: ' + str(total_bytes_received))
if __name__ == '__main__':
    print("File Transfer [Receiver]")
    avg_total_time = 0
    total_anticipated = 0
    p_list = []

    if len(sys.argv) < 4:
        print("Run with the following arguments: <expid> <send method> <list file name> <total run number>")
        exit(1)
    _exp_id = sys.argv[1]
    broadcast_name = sys.argv[3]
    _method = sys.argv[2]
    run_num = int(sys.argv[4])
    
    try:
        host_ip = os.getenv('IP_ADDR', None)
    except:
        print("ERROR: IP_ADDR variable not set")
        exit(1)

    os.system("taskset -p 0xff %d" % os.getpid())

    for run_idx in range (run_num):
        try:
            shutil.rmtree(broadcast_name[:-4])
        except OSError as e:
            print("Broadcast files in " + broadcast_name[:-4])
        os.mkdir(broadcast_name[:-4])
        print("Config Send Method: " + _method)
        if run_idx == 0:
            sock, sock_send = init(method=_method)
            avg_wait_times = [0] * len(files)
            total_written_bytes = [0] * len(files)
            for file in files:
                total_anticipated += file[1]
        print ("Total anticipated: " + str(total_anticipated))
        total_time = 0
        rec_buffer = Queue()
        start_time = Value('d', 0.0)
        end_time = Value('d', 0.0)
        file_received_events = []
        finish_times = Array('d', range(len(files)))
        for i in range(len(files)):
            event = Event()
            file_received_events.append(event)
        for i in range(NUM_RECEIVER_THREADS):
            receive_proc = Process(target=receive_thread, args=(file_received_events, sock[0], _method, start_time, ))
            p_list.append(receive_proc)
            receive_proc.start()
        write_proc = Process(target=count_bytes, args=(file_received_events, sock, sock_send, _method, end_time, finish_times))
        write_proc.start()
        write_proc.join()
        total_time = end_time.value - start_time.value
        #receive_files(sock, sock_send, _method)
        for p in p_list:
            p.terminate()
        print("App Running Time: " + str(total_time))
        
        avg_total_time += total_time
        for k in range(len(finish_times)):
            if k > 0:
                avg_wait_times[k] += (finish_times[k] - finish_times[k-1])
    
    for i in range(len(files)):
        avg_wait_times[i] = avg_wait_times[i] / run_num
    avg_total_time = avg_total_time / run_num
    
    results_path = RESULTS_BASE_DIR + os.getenv('CONTAINER_NAME', None)
    
    with open(results_path + "/out_" + _method + "_" + _exp_id + "_" + broadcast_name[:-4] + ".csv", 'wb') as file:
        fieldnames = ['name', 'wait_time']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(finish_times)):
            if i > 0:
                writer.writerow({
                    'name': files[i][0],
                    'wait_time': (avg_wait_times[i])
                })
        writer.writerow({
                    'name': 'App Runing Time',
                    'wait_time': avg_total_time
                })
                #print(files[i][0] + ": " + str(finish_times[i] - finish_times[i-1]))
    sys.exit(0)

