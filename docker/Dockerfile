FROM continuumio/miniconda3:latest

ARG uid
ARG username
ARG accelerator
ARG http_proxy
ARG https_proxy
ARG ssh_password

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

RUN apt-get update && apt-get install -y \
    "libgtk2.0-dev" \
    "libgl1-mesa-glx" \
    "vim" \
    "graphviz" \
	"gcc"

COPY ./conda /conda

RUN conda env update --file /conda/conda-env-${accelerator}.yml --name base

RUN useradd -m -s /bin/bash -u ${uid} --no-log-init ${username}

RUN echo "PS1='[\u@verres-environment \W]\$ '" >> /home/${username}/.bashrc

RUN mkdir /workspace && \
    mkdir /artifactory && \
    mkdir /data && \
    chmod -R a+rw /workspace && \
    chmod -R a+rw /artifactory && \
    chmod -R a+rw /data

COPY ./environment_setup.sh /etc/environment

# Setup SSH
RUN apt-get install -y openssh-server && \
    mkdir /var/run/sshd && \
    echo "${username}:${ssh_password}" | chpasswd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

COPY docker/entrypoint-ssh.sh /entrypoint-ssh

WORKDIR /workspace

CMD ["/bin/bash"]
