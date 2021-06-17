FROM intel/intel-optimized-tensorflow

ARG username=csa
ARG uid=1000

RUN useradd -m -s /bin/bash --no-log-init -u ${uid} ${username}

RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN pip3 install git+https://github.com/csxeba/Artifatorium.git

CMD ["/bin/bash"]
