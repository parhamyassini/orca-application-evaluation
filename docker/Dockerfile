FROM ubuntu

RUN apt-get update -y && \
    apt-get install -y python && \ 
    apt-get install -y net-tools

COPY ./broadcast_files_*.txt /home/
COPY ./client.py /home/client.py

WORKDIR /home

VOLUME /var/run/sockets
