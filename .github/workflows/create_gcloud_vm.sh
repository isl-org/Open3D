#!/usr/bin/env bash
set -euo pipefail

__usage="USAGE:
    $(basename $0) VM_NAME

    Creates a VM with the given name. Upon successful VM creation, the zone
    infomration will be written in /tmp/gcloud_vm_zone.txt.
"

OPEN3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. >/dev/null 2>&1 && pwd)"

print_usage_and_exit() {
    echo "$__usage"
    exit 1
}

if [[ "$#" -ne 1 ]]; then
    echo "Error: invalid number of arguments." >&2
    print_usage_and_exit
fi

VM_NAME="$1"
echo "Creating VM with name ${VM_NAME}"

GCE_ZONES=(us-west1-a
           us-west1-b
           us-central1-a
           us-central1-b
           us-central1-f
           us-east1-c
           us-east1-d
           us-east4-b
           southamerica-east1-c
           europe-west2-b
           europe-west3-b
           europe-west4-b
           europe-west4-c
           europe-west2-a
           asia-southeast1-b
           asia-southeast1-c
           australia-southeast1-a)

# https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus
# https://cloud.google.com/compute/docs/containers/deploying-containers
# docker pull gcr.io/open3d-dev/open3d-ubuntu-cuda-gcloud-ci:2-bionic-d90752d

VM_NAME="open3d-ubuntu-cuda-gcloud-ci-2-bionic-d90752d"
GCLOUD_DOCKER_TAG="gcr.io/open3d-dev/open3d-ubuntu-cuda-gcloud-ci:2-bionic-d90752d"
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
    --command "sudo docker run -d --rm ${GCLOUD_DOCKER_TAG} cat hello.txt"

gcloud compute instances delete ${VM_NAME} \
    --zone=${ZONE} \
    --quiet
