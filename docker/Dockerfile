FROM continuumio/miniconda3:latest

ARG uid
ARG accelerator=mkl

ENV PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y \
    "libgtk2.0-dev" \
    "libgl1-mesa-glx" \
    "vim" \
    "graphviz"

COPY ./conda /conda

RUN conda env update --file /conda/conda-env-${accelerator}.yml --name base

RUN useradd -m -s /bin/bash -u $uid --no-log-init user

RUN mkdir /workspace && \
    mkdir /artifactory && \
    mkdir /data && \
    chmod -R a+rw /workspace && \
    chmod -R a+rw /artifactory && \
    chmod -R a+rw /data

WORKDIR /workspace

CMD ["/bin/bash"]
