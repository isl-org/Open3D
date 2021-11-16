#!/usr/bin/env bash
#
# Use this script to create the base Google Cloud VM image for Ubuntu CUDA CI.
# This script only needs to be run once. It is recommended to run this script
# section by section, as it will prompt you for the necessary information.
#
# When we update the Google Cloud VM, change ubuntu-cuda.yml such that the
# "--image-family" points to the new image.
#
# List of packages installed:
# - Nvidia driver
# - Nvidia docker

set -euo pipefail

INSTANCE_NAME="open3d-ci-base-image-vm"
INSTANCE_ZONE="us-west1-a"
CUSTOM_IMAGE_NAME="ubuntu-os-docker-gpu-2004-20211116"

# Create VM.
gcloud compute instances create "${INSTANCE_NAME}" \
    --zone="${INSTANCE_ZONE}" \
    --service-account="open3d-ci-sa-gpu@open3d-dev.iam.gserviceaccount.com" \
    --accelerator="count=2,type=nvidia-tesla-t4" \
    --maintenance-policy="TERMINATE" \
    --machine-type="n1-standard-8" \
    --image-project="ubuntu-os-cloud" \
    --image-family="ubuntu-2004-lts" \
    --boot-disk-size="128GB" \
    --boot-disk-type="pd-ssd"
sleep 30

# Install latest nvidia-driver. This will not install CUDA toolkits and nvcc.
# Ref: https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
# This script may restart the macine. Run it twice.
gcloud compute ssh "${INSTANCE_NAME}" \
    --zone "${INSTANCE_ZONE}" \
    --command "curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py \
            && sudo python3 install_gpu_driver.py"
gcloud compute ssh "${INSTANCE_NAME}" \
    --zone "${INSTANCE_ZONE}" \
    --command "curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py \
            && sudo python3 install_gpu_driver.py"

# Check that nvidia-smi is working.
gcloud compute ssh "${INSTANCE_NAME}" \
    --zone "${INSTANCE_ZONE}" \
    --command "nvidia-smi"

# Install docker.
gcloud compute ssh "${INSTANCE_NAME}" \
    --zone "${INSTANCE_ZONE}" \
    --command "curl https://get.docker.com | sh \
            && sudo systemctl --now enable docker"

# Install nvidia-docker.
gcloud compute ssh "${INSTANCE_NAME}" \
    --zone "${INSTANCE_ZONE}" \
    --command "curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
            && curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
            && sudo apt-get update \
            && sudo apt-get install -y nvidia-docker2 \
            && sudo systemctl restart docker \
            && sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi"

# Turn off VM.
gcloud compute instances stop "${INSTANCE_NAME}" \
    --zone "${INSTANCE_ZONE}"

# Save image.
gcloud compute images create "${INSTANCE_NAME}" \
    --source-disk="${INSTANCE_NAME}" \
    --source-disk-zone="${INSTANCE_ZONE}" \
    --family="${CUSTOM_IMAGE_NAME}" \
    --storage-location=us

# Delete VM.
gcloud compute instances delete "${INSTANCE_NAME}" --zone="${INSTANCE_ZONE}"
