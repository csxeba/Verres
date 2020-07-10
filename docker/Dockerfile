FROM continuumio/miniconda3:latest

RUN conda install \
    "python>=3.7" \
    "tensorflow>=2.2" \
    "matplotlib"

RUN pip install opencv-python

CMD ["/bin/bash"]
