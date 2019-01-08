ARG REPOSITORY
ARG UBUNTU_VERSION

FROM ${REPOSITORY}/open3d-test:${UBUNTU_VERSION}-base

ARG UBUNTU_VERSION

COPY [ "/setup/setup-deps.sh", "./" ]

RUN /bin/bash /root/setup-deps.sh && rm /root/setup-deps.sh
