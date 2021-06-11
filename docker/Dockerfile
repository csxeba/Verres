FROM archlinux

ARG username=csa
ARG uid=1000

RUN pacman -Sy && \
    pacman -S --noconfirm \
    wget \
    sudo \
    fakeroot \
    binutils \
    pacman \
    yajl \
    base-devel

RUN useradd -m -s /bin/bash --no-log-init -u ${uid} ${username}
RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN mkdir -p /workspace
RUN chown -R ${username} /workspace

WORKDIR /workspace

COPY makepkg.conf /etc/makepkg.conf
USER ${username}

RUN wget https://aur.archlinux.org/cgit/aur.git/snapshot/package-query.tar.gz && \
RUN bash /workspace/getyaourt.bash && rm /workspace/getyaourt.bash

RUN yaourt -Sy && sudo yaourt -S --noconfirm --overwrite "*.py" --overwrite "*.pyc" --overwrite "*.pyo" \
    python \
    tensorflow-opt-native
