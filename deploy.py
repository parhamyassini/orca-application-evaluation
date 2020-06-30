import getpass
import sys
from fabric import Connection, Config

RESULTS_PATH = '/local-scratch/results'

DPDK_PATH_55 = '/home/pyassini/bess/bin/dpdk-devbind.py'
DPDK_PATH_56 = '/home/pyassini/bess/bin/dpdk-devbind.py'
DPDK_PATH_62 = '/local-scratch/bess/bin/dpdk-devbind.py'
DPDK_PATH_42 = '/local-scratch/bess/bin/dpdk-devbind.py'
DPDK_PATH_RAMSES = '/local-scratch/pyassini/bess/bin/dpdk-devbind.py'

DOCKER_PATH_55 = '/home/pyassini/docker'
DOCKER_PATH_56 = '/home/pyassini/docker'
DOCKER_PATH_62 = '/local-scratch/docker'
DOCKER_PATH_42 = '/local-scratch/docker'

IFACE_RAMSES_RACK_1 = 'ens6f0'
IFACE_RAMSES_RACK_2 = 'ens6f1'
IFACE_42 = 'enp1s0f0'
IFACE_62 = 'enp2s0f0'
IFACE_55 = 'enp1s0f0'
IFACE_56 = 'eth0'

DEV_RAMSES = '0000:81:00'
DEV_42 = '0000:01:00'
DEV_62 = '0000:02:00'
DEV_55 = '0000:01:00'
DEV_56 = '0000:01:00'

IP_BASE_42 = '192.168.0.33'
IP_BASE_62 = '192.168.0.65'
IP_BASE_55 = '192.168.1.33'
IP_BASE_56 = '192.168.1.65'

CONTAINER_NAME_BASE = 'container_'

def firewall_allow_iface(connection, iface_name):
    connection.sudo('iptables -I INPUT 1 -i ' + iface_name + ' -p all -j ACCEPT')

def nic_unbind(connection, dpdk_path, device_name):
    print ('Unbinding ' + device_name)
    connection.sudo(dpdk_path + ' --unbind ' + device_name + '.0')
    connection.sudo(dpdk_path + ' --unbind ' + device_name + '.1')

def nic_bind_uio(connection, dpdk_path, device_name):
    connection.sudo('modprobe uio_pci_generic')
    nic_unbind(connection, dpdk_path, device_name)
    connection.sudo(dpdk_path + ' -b uio_pci_generic ' + device_name + '.0')
    connection.sudo(dpdk_path + ' -b uio_pci_generic ' + device_name + '.1')

def nic_bind_kernel(connection, dpdk_path, device_name):
    connection.sudo(dpdk_path + ' -b ixgbe ' + device_name + '.0')
    connection.sudo(dpdk_path + ' -b ixgbe ' + device_name + '.1')

def iface_down(connection, iface_name):
    connection.sudo('ifdown ' + iface_name, hide='stderr')

def iface_up(connection, iface_name):
    connection.sudo('ifup ' + iface_name, hide='stderr')

def set_buffer_mem(connection):
    connection.sudo('sysctl -w net.core.rmem_max=2147483647')
    connection.sudo('sysctl -w net.core.rmem_default=2147483647')
    connection.sudo('sysctl -w net.core.wmem_max=2147483647')
    connection.sudo('sysctl -w net.core.wmem_default=2147483647')
    connection.sudo('sysctl -w net.unix.max_dgram_qlen=2147483647')

def switch_to_multicast(host_list):
    for host in host_list:
        if host['device_name'] != DEV_RAMSES:
            set_buffer_mem(host['connection'])
        # Deactivate the interfaces from kernel
        iface_down(host['connection'], host['iface_name_1'])
        if 'iface_name_2' in host:
            iface_down(host['connection'], host['iface_name_2'])
        nic_bind_uio(host['connection'], host['dpdk_path'], host['device_name'])
        
def switch_to_unicast(host_list):
    for host in host_list:
        if host['device_name'] != DEV_RAMSES:
            set_buffer_mem(host['connection'])
        nic_unbind(host['connection'], host['dpdk_path'], host['device_name'])
        nic_bind_kernel(host['connection'], host['dpdk_path'], host['device_name'])
        iface_up(host['connection'], host['iface_name_1'])
        firewall_allow_iface(host['connection'], host['iface_name_1'])
        if 'iface_name_2' in host:
            iface_up(host['connection'], host['iface_name_2'])
            firewall_allow_iface(host['connection'], host['iface_name_2'])

def setup_containers(host_list, num_containers, container_mode):
    for host in host_list:
        if host['device_name'] == DEV_RAMSES:
            continue
        connection = host['connection']
        for i in range(num_containers):
            local_address = int(host['ip_base'][10:]) + i
            container_ip = host['ip_base'][:10] + str(local_address)
            container_name = CONTAINER_NAME_BASE + str(i)
            print(container_name + ', ip: ' + container_ip)
            try:
                if (container_mode == 'unicast'):
                    connection.sudo(
                        'docker run -d --name ' + container_name 
                        + ' --cpuset-cpus ' + str(host['num_cpus'] - 1 - i)
                        + ' --net=pub_net --ip=' + container_ip 
                        + ' --env CONTAINER_NAME=' + container_name 
                        +' --env IP_ADDR=' + container_ip +
                        ' -v /var/run/sockets:/var/run/sockets receiver tail -f /dev/null')
                else:
                    connection.sudo(
                        'docker run -d --name ' + container_name
                        + ' --cpuset-cpus ' + str(host['num_cpus'] - 1 - i)
                        + ' --env CONTAINER_NAME=' + container_name 
                        +' --env IP_ADDR=' + container_ip +
                        ' -v /var/run/sockets:/var/run/sockets receiver tail -f /dev/null')
            except:
                print("Container setup failed (might be already up and running)")
                print("Skipped " + container_name + " with ip: " + container_ip)
                continue
            if host['num_cpus'] == 4: #nsl-42 is the old machine (an exception)
                break;

def run_receivers(host_list, num_containers, app_args):
    for host in host_list:
        if host['device_name'] == DEV_RAMSES:
            continue
        connection = host['connection']
        for i in range(num_containers):
            container_name = CONTAINER_NAME_BASE + str(i)
            connection.sudo('docker exec -d ' + container_name 
                + ' python client.py ' + app_args)

def remove_containers(host_list, num_containers):
    for host in host_list:
        if host['device_name'] == DEV_RAMSES:
            continue
        connection = host['connection']
        for i in range(num_containers):
            container_name = CONTAINER_NAME_BASE + str(i)
            try:
                connection.sudo('docker container stop ' + container_name)
                connection.sudo('docker container rm ' + container_name)
            except:
                print("Warning: Remove failed, container not exists")

def build_images(host_list):
    for host in host_list:
        if host['device_name'] == DEV_RAMSES:
            continue
        connection = host['connection']
        connection.sudo('docker image rm receiver')
        connection.run('cd ' + host['docker_path'])
        connection.sudo('docker build -t receiver .')

if __name__ == '__main__':
    print("Deployment Script")
    if len(sys.argv) < 3:
        print("Run with the following arguments: <username> <multicast/unicast>")
        exit(1)
    _username = sys.argv[1]    
    _goal = sys.argv[2]
    _container_mode = ""
    _num_containers = 0
    _experiment_name = ""
    if _goal == "setup_containers" or _goal == "remove_containers" or _goal == "run_receivers":
        try:
            _num_containers = int(sys.argv[4])
        except:
            print("Run with the following arguments:\
             <username> <setup/remove_containers/run_receivers> <unicast/multicast> <number of containers>")
        try:
            _container_mode = sys.argv[3]
        except:
            print("Run with the following arguments:\
             <username> <setup/remove_containers/run_receivers> <unicast/multicast> <number of containers>")

    if _goal == "run_receivers":
        try:
            _app_args = sys.argv[4]
        except:
            print("Run run_receivers with the following arguments:\
             <username> <run_receivers> <number of containers> <\"arguments for client.py\">")
    host_list = []
    
    sudo_pass = getpass.getpass("What's your sudo password?")
    config = Config(overrides={'sudo': {'password': sudo_pass}})
    print ("Connecting to ramses")
    ramses = Connection("localhost", port=22, user="pyassini", connect_kwargs={'password': sudo_pass}, config=config)
    print ("Connecting to nsl-42")
    nsl_42 = Connection("cs-nsl-42.cmpt.sfu.ca", port=22, user=_username, connect_kwargs={'password': sudo_pass}, config=config)
    print ("Connecting to nsl-62")
    nsl_62 = Connection("cs-nsl-62.cmpt.sfu.ca", port=22, user=_username, connect_kwargs={'password': sudo_pass}, config=config)
    print ("Connecting to nsl-55")
    nsl_55 = Connection("cs-nsl-55.cmpt.sfu.ca", port=22, user=_username, connect_kwargs={'password': sudo_pass}, config=config)
    print ("Connecting to nsl-56")
    nsl_56 = Connection("cs-nsl-56.cmpt.sfu.ca", port=22, user=_username, connect_kwargs={'password': sudo_pass}, config=config)
    ramses.sudo('whoami', hide='stderr')
    host_list.append({'connection': ramses,
    'dpdk_path': DPDK_PATH_RAMSES,
    'iface_name_1':IFACE_RAMSES_RACK_1,
    'iface_name_2': IFACE_RAMSES_RACK_2,
    'device_name':DEV_RAMSES,
    })
    host_list.append({'connection': nsl_42,
    'dpdk_path': DPDK_PATH_42,
    'docker_path': DOCKER_PATH_42,
    'iface_name_1':IFACE_42,
    'device_name':DEV_42,
    'ip_base': IP_BASE_42,
    'num_cpus': 4
    })
    host_list.append({'connection': nsl_62,
    'dpdk_path': DPDK_PATH_62,
    'docker_path': DOCKER_PATH_62,
    'iface_name_1':IFACE_62,
    'device_name':DEV_62,
    'ip_base': IP_BASE_62,
    'num_cpus': 16
    })
    host_list.append({'connection': nsl_55,
    'dpdk_path': DPDK_PATH_55,
    'docker_path': DOCKER_PATH_55,
    'iface_name_1':IFACE_55,
    'device_name':DEV_55,
    'ip_base': IP_BASE_55,
    'num_cpus': 12
    })
    host_list.append({'connection': nsl_56,
    'dpdk_path': DPDK_PATH_56,
    'docker_path': DOCKER_PATH_56,
    'iface_name_1':IFACE_56,
    'device_name':DEV_56,
    'ip_base': IP_BASE_56,
    'num_cpus': 12
    })
    print (_goal)
    
    if (_goal == 'multicast'):
        switch_to_multicast(host_list)
        print("\nSuccesfully done multicast configuration")
    elif (_goal == 'unicast'):
        switch_to_unicast(host_list)
        print("\nSuccesfully done unicast configuration")
    elif (_goal == 'setup_containers'):
        setup_containers(host_list, _num_containers, _container_mode)
        print ("\nSuccesfully setup containers")
    elif (_goal == 'remove_containers'):
        print("Warnining: removing containers!")
        remove_containers(host_list, _num_containers)
        print ("\nSuccesfully removed all containers")
    elif(_goal=='run_receivers'):
        print("Running " + str(_num_containers) + " apps on each receiver")
        run_receivers(host_list, _num_containers, _app_args)
        print ("\nSuccesfully run apps in containers")
    elif (_goal == 'build_images'):
        build_images(host_list)
        print ("\nSuccesfully built images")

