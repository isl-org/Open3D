ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /root

COPY [ "/setup/googletest-release-1.8.0.tar.gz", \
       "/setup/install-gtest.sh", \
       "/setup/setup-base.sh", \
       "/setup/test.sh", \
       "/setup/.bashrc", \
       "./" ]

RUN /bin/bash /root/setup-base.sh && rm /root/setup-base.sh
