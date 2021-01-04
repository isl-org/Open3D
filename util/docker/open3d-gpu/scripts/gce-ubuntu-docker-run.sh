#!/usr/bin/env bash

## Shell scripts for GCE CI workflow, called from the github actions yml file
## (for portability and ease of maintenance)

# exit when any command fails
set -e

## These variables should be in the environment (from github repo secrets)
# GCE_PROJECT
# GCE_GPU_CI_SA
## GITHUB environment
# GITHUB_SHA
## CI test matrix:
CI_CONFIG_ID=${CI_CONFIG_ID:=0}
CI_CONFIG_ID=${CI_CONFIG_ID%%-*} # '3-ML-bionic' -> '3'

# CI configuration specification
SHARED=(OFF ON OFF ON OFF OFF)
BUILD_ML_OPS=(OFF ON OFF ON ON ON)
BUILD_CUDA_MODULE=(OFF OFF ON ON ON ON)
BUILD_RPC_INTERFACE=(ON ON OFF OFF ON ON)
UBUNTU_VERSION_LIST=(bionic bionic bionic bionic bionic focal)
UBUNTU_VERSION=${UBUNTU_VERSION:-${UBUNTU_VERSION_LIST[$CI_CONFIG_ID]}}
BUILD_TENSORFLOW_OPS=("${BUILD_ML_OPS[@]}")
BUILD_PYTORCH_OPS=("${BUILD_ML_OPS[@]}")

# VM instance configuration
NPROC=8                            # {2,4,8,16,32,64,96}
GCE_INSTANCE_BASE_TYPE=n1-standard # GCE only allows n1-standard machines with GPUs (2020/07)
GCE_INSTANCE_TYPE=${GCE_INSTANCE_BASE_TYPE}-$NPROC
GCE_INSTANCE_BASENAME=ci-gpu-vm
GCE_INSTANCE=${GCE_INSTANCE_BASENAME}-${GITHUB_SHA::8}-${CI_CONFIG_ID}
# See https://cloud.google.com/compute/docs/gpus for GCE zones supporting Tesla
# T4 GPus
GCE_INSTANCE_ZONE=(us-west1-a us-west1-b
    us-central1-a us-central1-b us-central1-f
    us-east1-c us-east1-d us-east4-b
    southamerica-east1-c
    europe-west2-b europe-west3-b europe-west4-b europe-west4-c europe-west2-a
    asia-southeast1-b asia-southeast1-c australia-southeast1-a
)

GCE_ZID=${GCE_ZID:=0} # Persist between calls of this script
GCE_GPU="count=1,type=nvidia-tesla-t4"
GCE_BOOT_DISK_TYPE=pd-ssd
GCE_BOOT_DISK_SIZE=32GB
NVIDIA_DRIVER_VERSION=455 # Must be present in Ubuntu repos 20.04: {390, 418, 430, 435, 440, 450, 455}
GCE_VM_BASE_OS=ubuntu20.04
GCE_VM_IMAGE_SPEC=(--image-project=ubuntu-os-cloud --image-family=ubuntu-2004-lts)
GCE_VM_CUSTOM_IMAGE_FAMILY=ubuntu-os-docker-gpu-2004-lts
GCE_VM_CUSTOM_IMAGE=open3d-gpu-ci-base-20201228
VM_IMAGE=open3d-gpu-ci-base-$(date +%Y%m%d)
GCE_CI_TIMEOUT=5400 # Self delete VM after timeout (seconds)

# Container configuration
REGISTRY_HOSTNAME=gcr.io
DC_IMAGE="$REGISTRY_HOSTNAME/$GCE_PROJECT/open3d-gpu-ci-$UBUNTU_VERSION"
DC_IMAGE_TAG="$DC_IMAGE:$GITHUB_SHA"
DC_IMAGE_LATEST_TAG="$DC_IMAGE:latest"

case "$1" in

# Set up docker to authenticate via gcloud command-line tool.
gcloud-setup)
    gcloud auth configure-docker
    gcloud info
    ;;

    # Build the Docker image
docker-build)
    # Pull previous image as cache
    docker pull "$DC_IMAGE_LATEST_TAG" || true
    DOCKER_BUILDKIT=1 docker build -t "$DC_IMAGE_TAG" \
        -f util/docker/open3d-gpu/Dockerfile \
        --build-arg UBUNTU_VERSION="$UBUNTU_VERSION" \
        --build-arg NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION}" \
        .
    docker tag "$DC_IMAGE_TAG" "$DC_IMAGE_LATEST_TAG"
    ;;

    # Push the Docker image to Google Container Registry
docker-push)
    docker push "$DC_IMAGE_TAG"
    docker push "$DC_IMAGE_LATEST_TAG"
    ;;

create-base-vm-image)
    gcloud compute instances create "$VM_IMAGE" \
        --zone="${GCE_INSTANCE_ZONE[$GCE_ZID]}" \
        --service-account="$GCE_GPU_CI_SA" \
        --accelerator="$GCE_GPU" \
        --maintenance-policy=TERMINATE \
        --machine-type=$GCE_INSTANCE_TYPE \
        "${GCE_VM_IMAGE_SPEC[@]}" \
        --boot-disk-size=$GCE_BOOT_DISK_SIZE \
        --boot-disk-type=$GCE_BOOT_DISK_TYPE \
        --metadata=startup-script="\
        sudo apt-key adv --fetch-keys https://nvidia.github.io/nvidia-docker/gpgkey; \
        curl -s -L https://nvidia.github.io/nvidia-docker/$GCE_VM_BASE_OS/nvidia-docker.list \
        | sudo tee /etc/apt/sources.list.d/nvidia-docker.list; \
        sudo apt-get update; \
        sudo apt-get -y upgrade; \
        sudo apt-get -y install nvidia-driver-${NVIDIA_DRIVER_VERSION} nvidia-container-toolkit docker.io; \
        sudo apt-get -y autoremove; \
        sudo systemctl enable docker; \
        sudo gcloud auth configure-docker; \
        sudo poweroff"
    echo "Waiting 5 minutes for VM image creation script..."
    sleep 300
    TRIES=0
    until ((TRIES >= 5)) || gcloud compute images create "$VM_IMAGE" --source-disk="$VM_IMAGE" \
        --source-disk-zone="${GCE_INSTANCE_ZONE[$GCE_ZID]}" \
        --family="$GCE_VM_CUSTOM_IMAGE_FAMILY" \
        --storage-location=us; do
        sleep 60
        ((TRIES = TRIES + 1))
    done
    gcloud compute instances delete "$VM_IMAGE" --zone="${GCE_INSTANCE_ZONE[$GCE_ZID]}"
    ;;

create-vm)
    # Try creating a VM instance in each zone
    until
        ((GCE_ZID >= ${#GCE_INSTANCE_ZONE[@]})) ||
            gcloud compute instances create "$GCE_INSTANCE" \
                --zone="${GCE_INSTANCE_ZONE[$GCE_ZID]}" \
                --accelerator="$GCE_GPU" \
                --maintenance-policy=TERMINATE \
                --machine-type=$GCE_INSTANCE_TYPE \
                --boot-disk-type=$GCE_BOOT_DISK_TYPE \
                --image="$GCE_VM_CUSTOM_IMAGE" \
                --service-account="$GCE_GPU_CI_SA" \
                --scopes=default,compute-rw \
                --metadata=startup-script="\
                sleep ${GCE_CI_TIMEOUT};\
                gcloud --quiet compute instances delete ${GCE_INSTANCE} \
                --zone=${GCE_INSTANCE_ZONE[$GCE_ZID]}"
    do
        ((GCE_ZID = GCE_ZID + 1))
    done
    sleep 30 # wait for instance ssh service startup
    export GCE_ZID
    echo "GCE_ZID=$GCE_ZID" >>"$GITHUB_ENV"       # Export environment variable for next step
    exit $((GCE_ZID >= ${#GCE_INSTANCE_ZONE[@]})) # 0 => success
    ;;

run-ci)
    gcloud compute ssh "${GCE_INSTANCE}" --zone "${GCE_INSTANCE_ZONE[$GCE_ZID]}" --command \
        "sudo docker run --detach --interactive --name open3d_gpu_ci --gpus all \
            --env NPROC=$NPROC \
            --env SHARED=${SHARED[$CI_CONFIG_ID]} \
            --env BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE[$CI_CONFIG_ID]} \
            --env BUILD_RPC_INTERFACE=${BUILD_RPC_INTERFACE[$CI_CONFIG_ID]} \
            --env BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS[$CI_CONFIG_ID]} \
            --env BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS[$CI_CONFIG_ID]} \
            --env OPEN3D_ML_ROOT=/root/Open3D/Open3D-ML \
            $DC_IMAGE_TAG; \
            sudo docker exec --interactive  open3d_gpu_ci util/run_ci.sh"
    ;;

run-python-ci)
    # Wait for 60 mins wheel to be downloaded from ubuntu.yml CI
    gcloud compute ssh "${GCE_INSTANCE}" --zone "${GCE_INSTANCE_ZONE[$GCE_ZID]}" \
        <<"WHEEL_WAIT"
    bash -o errexit -o nounset -o pipefail -o xtrace -c 'for wait_time in $(seq 0 60); do
        whlstat="$(ls -l --full-time ~/open3d*.whl 2>/dev/null)"
        sleep 60
        if [ -n "$whlstat" ] && \
        [[ "$(ls -l --full-time ~/open3d*.whl)" == "$whlstat" ]]; then
            echo Wheel available!
            wait_time=-1
            break
        fi
        echo Waiting for wheel since ${wait_time} min
    done
    if ((wait_time>-1)); then
        echo Timeout: No Python wheel available for GPU tests.
        exit 1
    else
        sudo docker cp ~/open3d*.whl open3d_gpu_ci:/root/
        echo Run Python CI
        sudo docker exec --interactive open3d_gpu_ci \
            bash -o errexit -o nounset -o pipefail -o xtrace -c \
            "source util/ci_utils.sh; test_wheel /root/open3d*.whl; run_python_tests"
    fi'
WHEEL_WAIT
    ;;

delete-image)
    gcloud container images untag "$DC_IMAGE_TAG" --quiet
    # Clean up images without tags - keep :latest
    gcloud container images list-tags "$DC_IMAGE" --filter='-tags:*' \
        --format='get(digest)' --limit=unlimited |
        xargs -I "{arg}" gcloud container images delete "${DC_IMAGE}@{arg}" --quiet
    ;;

delete-vm)
    gcloud compute instances delete "${GCE_INSTANCE}" --zone "${GCE_INSTANCE_ZONE[$GCE_ZID]}"
    ;;

ssh-vm)
    gcloud compute ssh "${GCE_INSTANCE}" --zone "${GCE_INSTANCE_ZONE[$GCE_ZID]}"
    ;;

*)
    echo "Error: Unknown sub-command $1"
    ;;
esac
