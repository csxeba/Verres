FROM continuumio/miniconda3:latest

ARG uid

ENV PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y \
    "libgtk2.0-dev" \
    "libgl1-mesa-glx" \
    "vim" \
    "graphviz"

RUN conda install \
    "python>=3.7" \
    "tensorflow>=2.2" \
    "matplotlib"

RUN pip install \
    "opencv-python" \
    "pydot-ng" \
    "pycocotools-fix" \
    "git+https://github.com/csxeba/Artifactorium.git"

RUN useradd -m -s /bin/bash -u $uid --no-log-init user

RUN mkdir /workspace && \
    chmod -R a+rw /workspace

WORKDIR /workspace

CMD ["/bin/bash"]
