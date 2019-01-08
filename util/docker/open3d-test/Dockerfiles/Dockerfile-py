ARG REPOSITORY
ARG UBUNTU_VERSION
ARG BUNDLE_TYPE

FROM ${REPOSITORY}/open3d-test:${UBUNTU_VERSION}-${BUNDLE_TYPE}

ARG PYTHON

COPY [ "/setup/setup-py.sh", "./" ]

RUN /root/setup-py.sh ${PYTHON} && rm /root/setup-py.sh
