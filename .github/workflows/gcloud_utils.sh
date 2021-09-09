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

# Create container
# --service-account="$GCE_GPU_CI_SA" \
gcloud compute instances create \
    open3d-ubuntu-cuda-gcloud-ci-2-bionic-d90752d \
    --project open3d-dev \
    --no-service-account --no-scopes \
    --image-family common-cu110 \
    --image-project deeplearning-platform-release \
    --zone=us-east1-c \
    --accelerator="count=2,type=nvidia-tesla-t4" \
    --maintenance-policy=TERMINATE \
    --machine-type=n1-standard-4 \
    --boot-disk-type=pd-ssd

# gcloud compute ssh \
#     open3d-ubuntu-cuda-gcloud-ci-2-bionic-d90752d \
#     --zone=us-east1-c \
#     --command "nvidia-smi"

# gcloud compute ssh "${GCE_INSTANCE}" --zone "${GCE_INSTANCE_ZONE[$GCE_ZID]}" --command \
#     "sudo docker run --detach --interactive --name open3d_gpu_ci --gpus all \
#         --env NPROC=$NPROC \
#         --env SHARED=${SHARED[$CI_CONFIG_ID]} \
#         --env BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE[$CI_CONFIG_ID]} \
#         --env BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS[$CI_CONFIG_ID]} \
#         --env BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS[$CI_CONFIG_ID]} \
#         --env OPEN3D_ML_ROOT=/root/Open3D/Open3D-ML \
#         $DC_IMAGE_TAG; \
#         sudo docker exec --interactive  open3d_gpu_ci util/run_ci.sh"
# ;;
