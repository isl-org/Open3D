#!/usr/bin/env bash
set -euo pipefail

# __usage="USAGE:
#     $(basename $0) [ACTION]

# OPTION:
#     create_instance    : Build with OpenBLAS x86_64
#     delete_instance    : Build with OpenBLAS ARM64
# "

# OPEN3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. >/dev/null 2>&1 && pwd)"
# GIT_HASH="$(git rev-parse --short HEAD)"

# print_usage_and_exit() {
#     echo "$__usage"
#     exit 1
# }

# echo "[$(basename $0)] command $1"
# case "$1" in
#     create_instance)
#         echo "create_instance"
#         exit 0
#         ;;
#     delete_instance)
#         echo "delete_instance"
#         exit 0
#         ;;
#     *)
#         echo "Error: invalid argument: ${1}." >&2
#         print_usage_and_exit
#         ;;
# esac


# https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus
# https://cloud.google.com/compute/docs/containers/deploying-containers
# docker pull gcr.io/open3d-dev/open3d-ubuntu-cuda-gcloud-ci:2-bionic-d90752d

VM_NAME="open3d-ubuntu-cuda-gcloud-ci-2-bionic-d90752d"
GCE_DOCKER_TAG="gcr.io/open3d-dev/open3d-ubuntu-cuda-gcloud-ci:2-bionic-d90752d"
ZONE=australia-southeast1-a

# Create container
# --no-service-account --no-scopes \
gcloud compute instances create ${VM_NAME} \
    --project open3d-dev \
    --service-account="$GCE_GPU_CI_SA" \
    --image-family common-cu110 \
    --image-project deeplearning-platform-release \
    --zone=${ZONE} \
    --accelerator="count=2,type=nvidia-tesla-t4" \
    --maintenance-policy=TERMINATE \
    --machine-type=n1-standard-4 \
    --metadata "install-nvidia-driver=True,proxy-mode=project_editors" \
    --boot-disk-type=pd-ssd

# Nvidia-driver takes about 1 minute to install in the background
sleep 90s

gcloud compute ssh ${VM_NAME} \
    --zone=${ZONE} \
    --command "nvidia-smi"

gcloud compute ssh ${VM_NAME} \
    --zone=${ZONE} \
    --command "sudo docker run -d --rm ${GCE_DOCKER_TAG} cat hello.txt"

gcloud compute instances delete ${VM_NAME} \
    --zone=${ZONE} \
    --quiet
