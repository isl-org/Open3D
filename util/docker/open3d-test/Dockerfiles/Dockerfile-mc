ARG REPOSITORY
ARG UBUNTU_VERSION
ARG BUNDLE_TYPE

FROM ${REPOSITORY}/open3d-test:${UBUNTU_VERSION}-${BUNDLE_TYPE}

ARG MC_INSTALLER
ARG CONDA_DIR

COPY [ "/setup/${MC_INSTALLER}", "/setup/setup-mc.sh", "./" ]

RUN /root/setup-mc.sh ${MC_INSTALLER} ${CONDA_DIR} && rm /root/setup-mc.sh
