This is the repository for application layer codes for end-to-end evaluation of Orca multicast system

# Experiment Setup
The report.pdf file in the docs folder contains detailed description of the experiments and implementation details. 
Important figures also included here:
[figures]

# Codes and Dependencies
### Python Libraries
The file *requirements.txt* provides the necessary libraries for running python scripts.
> The application codes were tested using *Python 2.7.12*.

### BESS Codes
Orca agent implementation on hosts requires BESS software switch. 
The modified version of BESS repo containing Orca modules is available on [this repository](https://github.com/khaledmdiab/bess/tree/dev-parham).
 > Note that current working version is on the "dev-parham" branch.

For installation follow the [the wiki page](https://github.com/NetSys/bess/wiki/Build-and-Install-BESS) instruction while replacing the cloning repo with the provided repository above.

In the last step for building BESS add the plugin option to compile orca module codes:
```
./build.py --plugin mdc_receiver_plugin
```
### Docker
Docker should be installed on all of the receiver devices. 
For ubuntu docker installation follow [this guide](https://docs.docker.com/engine/install/ubuntu/#installation-methods).
> Docker version 19.03.9.

## Organization
### Codes
Codes directory contains python scripts for server and client applications. 
The "docker" sub-directory, contains the image for running multiple clients inside one host (check next section for instructions).

### deploy.py
The deploy.py script uses Python *Fabric* for performing the necessary steps on the receiver machines from a centeralized machine. This code should be located at the server machine (currently cs-nsl-ramses).

### plot_results.py
This script is used for plotting the result of the experiment. The output format and selecting figures are handled inside the code.

# Configuration Instructions 
These are the steps to be done before running experiments. Ideally, these seteps need to be done only once. 
## Hugepages
Configure HugePages should be done after startup. Ideally, you should do this step as soon as possible to avoid fragmantation.

On Ramses: 
```
cd <bess directory>
sudo ./start_huge_pages_numa.sh
```

On receivers:
```
sudo ./start_huge_pages_single.sh
```
> **Important note:** For the constrained version of Orca we need 4096 huge pages  (with 2MB size). So for the receivers  huge page reservation  script shoudl be changed to : "sysctl vm.nr_hugepages=4096"

## Configure network interfaces
For the unicast setup, we assign static IPs for the 10G network interface of the machines. 
On each machine, edit the ```/etc/network/interfaces``` file and add the following:

**cs-nsl-ramses**: 
```
iface ens6f1 inet static
address 192.168.0.1
netmask 255.255.255.0
network 192.168.0.0
broadcast 192.168.0.255

iface ens6f0 inet static
address 192.168.1.1
netmask 255.255.255.0
network 192.168.1.0
broadcast 192.168.1.255
```

**cs-nsl-56:**
```
iface eth0 inet static
address 192.168.1.3
netmask 255.255.255.0
network 192.168.1.0
broadcast 192.168.1.255
```

**cs-nsl-55:**
```
iface enp1s0f0 inet static
address 192.168.1.2
netmask 255.255.255.0
network 192.168.1.0
broadcast 192.168.1.255
```
**cs-nsl-62:**
```
iface enp2s0f0 inet static
address 192.168.0.3
netmask 255.255.255.0
network 192.168.0.0
broadcast 192.168.0.255
```

**cs-nsl-42:**
```
iface enp1s0f0 inet static
address 192.168.0.4
netmask 255.255.255.0
network 192.168.0.0
broadcast 192.168.0.255
```

## Configure docker network
For the unicast setup, the containers use "macvlan" docker network which enables the containers to share the physical NIC by having dedicated "virtual" IP and MAC addresses. So that from the switch point of view, each containerizing is transparent and containers are treated as  physical machine whith IP and MAC.

> Note that this setup needs the NIC to be binded to Kernel so it can be done only when switched to the unicast setup. For unicast setup instruction see the next section Run Instructions.

We devide the subnets of each network (for each switch), to 8 networks of 32 IPs and assign these IPs to containers of each machine. (--ip-range parameter). IPs x.x.x.2, x.x.x.3, x.x.x.4 ... are assigned to the physical machines so for the containers we start with the x.x.x.32/27 IPs.

**cs-nsl-55:**
```
sudo docker network create -d macvlan \
--subnet=192.168.1.0/24 \
--ip-range=192.168.1.32/27 \
--gateway=192.168.1.32 \
-o parent=enp1s0f0 pub_net
``` 

**cs-nsl-56:**
```
sudo docker network create -d macvlan \
--subnet=192.168.1.0/24 \
--ip-range=192.168.1.64/27 \
--gateway=192.168.1.64 \
-o parent=eth0 pub_net
``` 
**cs-nsl-42**:
```
sudo docker network create -d macvlan \
--subnet=192.168.0.0/24 \
--ip-range=192.168.0.32/27 \
--gateway=192.168.0.64 \
-o parent=enp1s0f0 pub_net
```
**cs-nsl-62**:
```
sudo docker network create -d macvlan \
--subnet=192.168.0.0/24 \
--ip-range=192.168.0.64/27 \
--gateway=192.168.0.64 \
-o parent=enp2s0f0 pub_net
```
## Programming NetFPGAs
Before running the experiments, NetFGPA (NF) should be programmed.
For the multicast evaluations the NF switches should be programmed with the Orca bitfile and for unicast it is programmed with reference switch (L2 switch) bitfile.

> Current NF Hosts: cs-nsl-57, cs-nsl-58

### Orca Switch 
Log in to the NF hosts and on each host execute:
```
cd /local-scratch/kdiab/Projects/netfpga-sume-live/projects/reference_mdc/bitfiles/
sudo -E env "PATH=$PATH" xmd
```
Inside the XMD CLI:
``` fpga -f reference_orca.bit```

And then reboot the system before using the new NF switch.

### Reference (L2 Switch )
```
cd /local-scratch/kdiab/Projects/netfpga-sume-live/projects/reference_switch/bitfiles/

sudo -E env "PATH=$PATH" xmd
```
Inside the XMD CLI:
``` fpga -f reference_switch.bit```

## Make the shared directory
We share a path between the host and docker containers to enable unix domain socket communication. For using the default path (written in the dockerfile and client.py app) make this directory in each machine:

```
mkdir /var/run/sockets/
```

## Build docker image 
After cloning this repository, on each receiver machine: 
```
cd orca-application-evaluation/docker
sudo docker build -t receiver .
```

## Setting the environment variables
The BESS script will use these environment variables to know how many containers are running and the name of the containers. For each experiment set the desired value by adding these lines to ```~/.bashrc```: 
```
export NUM_INSTANCES=2
export CONTAINER_NAME_BASE="container_"
```
then, run ``` source ~/.bashrc```.

# Run Instructions

## Deploy
The deploy.py script is used for configuring the machines in the testbed for multicast/unicast setup and managing the containers from the centeralized point.
Currently cs-nsl-ramses is used as the server:
```
/local-scratch/pyassini/orca-application-evaluation
```
For each command the script will prompt for sudo password.
> The code is not reusable yet and has some constant values (e.g docker path and interface names). Checkout the hardcoded parameters on the script if any hardware or path changes have made.

### 1.a Setup Multicast
This command will handle the necessary steps for unbinding the NIC (for kernel bypassing) and configuring buffer sizes.
```
python deploy.py <username> multicast
```
### 1.b Setup Unicast 
This command will bind the NIC to the Kernel and assign the IP addresses as configured in the previous steps.
```
python deploy.py <username> unicast
```
### 2. Build Docker Image
This command can be skipped if you have manually built images in receiver machines in the configuration step.
```
python deploy.py <username> build_images
```

### 3. Setup Containers
This command will perform the necessary steps for setting up and running the containers on machines. 
```
python deploy.py <username> setup_containers <multicast/unicast> <number of containers>
```
> Note that cs-nsl-42 has only 4 cores and we run only one instance on this machine, it is hard coded (temporarily) in the deploy.py script.

### Removing containers
Containers should be removed before switching between unicast and multicast.
```
python deploy.py <username> setup_containers <number of containers>
```

## Run Experiments
Running starts with client apps and then running the server app. 
For the multicast experiments orca agent should be running on server and receiver machines. To run the orca agent:
```
cd /local-scratch/pyassini/bess
sudo -E bessctl/bessctl
```
Inside the BESS CLI of server, enter ```run samples/orca_agent_send```.

In the CLI of clients enter ```run samples/orca_agent_receive```.
> An alternative version of script is also added as "orca_agent_receive_constrained" which only uses constant 3 CPU cores regardless of number of containers.

### Client Script
Clients can be run manually using the interactive shell of containers:
```
sudo docker exec -it <container_name> python client.py <client args>
```
Alternetavily they can be run using deploy.py by specifying number of instances to be executed on each machine:

```
python deploy.py <username> run_receivers <number of instances> <"client args"> 
```

**Arguments:**
```
<experiment_name> <method> <workload_files_list> <total_runs>
```

* experiment_name: Used for generating outputs, in our convention it has the number of total receivers after letter "n" and number of concurrent send processes after "m" (only in unicast case). Example: "n12m6" for unicast and "n12" for multicast.

* method: it can be either "udp" or "orca".
* workload_files_list: name of the file containing the list of the files and sizes to be received. Example: "broadcast_files_n.txt" where n can be from 0 to 4.
> These files are pre-generated to make the client aware of file size that should be received (for simplicity). See next section for adding custome workloads.
* total_runs: This argument specifies the anticipated rounds for experiment. The results file generated will contain average time of these experiments.

 Examples:
```
sudo docker exec -it container_3 python client.py n12m6 udp broadcast_files_4.txt 2
```
```
python deploy.py pyassini run_receivers 2 "n8 orca broadcast_files_3.txt 10"
```

### Server Script
This script by default should be executed in the parent directory of the workloads. The workloads are generated using random generator based on the size of the spark example of LDA application. Each workload is then doubled (broadcast_files 0 to 4).

The IP address of the receivers is hardcoded inside the server.py script (different experiments are commented out). Make sure to set the correct addresses before running experiments. 

```
python server.py <args>
```

**Arguments**:
```
<method> <workload_directory_name> <total_runs> <number_of_send_procs>
```

* method: Can be "udp" or "orca".
* workload_directory_name: Name of the folder containing the the files. Example: “broadcast_files_n” where n can be from 0 to 4.

* total_runs: Same number as the receiver argument, determines number of times an experiment should be runned before generating results.

* number_of_send_procs: In UDP, this input will determine the number of processes for concurrent sending to the destinations. 
> Number of receivers should be divisible by the number of processes.

Example:
```
sudo python server.py orca broadcast_files_0 5 1
```
```
sudo python server.py udp broadcast_files_3 4 6
```
## Collecting Results
The server results for "App execution time" and send time of each file are stored in the specified folder (by default "'../results/ramses/'") in server.py script {line 13}.
Results for each receiver inside the containers is stored in the directory with this format: 
```
/var/run/sockets/results/container_*/
```

## Plot Results


## Notes and Known Issues
