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

# CI configuration specification
BUILD_DEPENDENCY_FROM_SOURCE=( ON ON ON ON ON )
SHARED=( OFF ON OFF ON OFF )
BUILD_ML_OPS=( OFF ON OFF ON ON )
BUILD_TENSORFLOW_OPS=( "${BUILD_ML_OPS[@]}" )
BUILD_PYTORCH_OPS=( "${BUILD_ML_OPS[@]}" )
BUILD_CUDA_MODULE=( OFF OFF ON ON ON )
BUILD_RPC_INTERFACE=( ON ON OFF OFF ON )

# VM instance configuration
NPROC=8                      # {2,4,8,16,32,64,96}
GCE_INSTANCE_BASE_TYPE=n1-standard      # GCE only allows n1-standard machines with GPUs (2020/07)
GCE_INSTANCE_TYPE=${GCE_INSTANCE_BASE_TYPE}-$NPROC
GCE_INSTANCE_BASENAME=ci-gpu-vm
GCE_INSTANCE=${GCE_INSTANCE_BASENAME}-${GITHUB_SHA::8}-${CI_CONFIG_ID}
GCE_INSTANCE_ZONE=us-west1-b
GCE_GPU="count=1,type=nvidia-tesla-t4"
GCE_BOOT_DISK_TYPE=pd-ssd
GCE_BOOT_DISK_SIZE=32GB
NVIDIA_DRIVER_VERSION=440    # Must be present in Ubuntu repos
CUDA_VERSION=10.1
CUDNN="cudnn7-"             # {"", "cudnn7-", "cudnn8-"}
GCE_VM_BASE_OS=ubuntu20.04
GCE_VM_IMAGE_SPEC="--image-project=ubuntu-os-cloud --image-family=ubuntu-2004-lts"
VM_IMAGE=open3d-gpu-ci-base-$(date +%Y%m%d)

# Container configuration
CONTAINER_BASE_OS=ubuntu18.04
REGISTRY_HOSTNAME=gcr.io
DC_IMAGE_TAG="$REGISTRY_HOSTNAME/$GCE_PROJECT/open3d-gpu-ci-base:$GITHUB_SHA"


case "$1" in

        # Set up docker to authenticate via gcloud command-line tool.
    gcloud-setup )
        gcloud auth configure-docker
        gcloud info
        ;;

      # Build the Docker image
    docker-build )
        docker build -t "$DC_IMAGE_TAG" \
            -f util/docker/open3d-gpu/Dockerfile \
            --build-arg CUDA_VERSION="$CUDA_VERSION" \
            --build-arg CONTAINER_BASE_OS="$CONTAINER_BASE_OS" \
            --build-arg CUDNN="$CUDNN" .
        ;;

      # Push the Docker image to Google Container Registry
    docker-push )
        docker push "$DC_IMAGE_TAG"
        ;;


    create-base-vm-image )
        gcloud compute instances create $VM_IMAGE \
        --zone=${GCE_INSTANCE_ZONE} \
        --service-account="$GCE_GPU_CI_SA" \
        --accelerator="$GCE_GPU" \
        --maintenance-policy=TERMINATE \
        --machine-type=$GCE_INSTANCE_TYPE \
        $GCE_VM_IMAGE_SPEC \
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
        until (( TRIES>=5 )) || gcloud compute images create ${VM_IMAGE} --source-disk=$VM_IMAGE \
            --source-disk-zone=${GCE_INSTANCE_ZONE} \
            --family=ubuntu-os-docker-gpu-1804-lts \
            --storage-location=us ; do
            sleep 60
            (( TRIES=TRIES+1 ))
        done
        gcloud compute instances delete $VM_IMAGE --zone=${GCE_INSTANCE_ZONE}
        ;;

    create-vm )
        TRIES=0       # Try creating a VM instance 5 times
        until (( TRIES>=5 )) || \
            gcloud compute instances create $GCE_INSTANCE \
            --zone=${GCE_INSTANCE_ZONE} \
            --accelerator="$GCE_GPU" \
            --maintenance-policy=TERMINATE \
            --machine-type=$GCE_INSTANCE_TYPE \
            --boot-disk-size=$GCE_BOOT_DISK_SIZE \
            --boot-disk-type=$GCE_BOOT_DISK_TYPE \
            --image-family=ubuntu-os-docker-gpu-1804-lts \
            --service-account="$GCE_GPU_CI_SA"
            do
                sleep 5
                (( TRIES=TRIES+1 ))
            done
            sleep 60              # wait for instance startup
            exit $(( TRIES>=5 ))  # 0 => success
            ;;

    run-ci )
        gcloud compute ssh ${GCE_INSTANCE} --zone ${GCE_INSTANCE_ZONE} --command \
            "sudo docker run --rm --gpus all \
            --env NPROC=$NPROC \
            --env SHARED=${SHARED[$CI_CONFIG_ID]} \
            --env BUILD_DEPENDENCY_FROM_SOURCE=${BUILD_DEPENDENCY_FROM_SOURCE[$CI_CONFIG_ID]} \
            --env BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE[$CI_CONFIG_ID]} \
            --env BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS[$CI_CONFIG_ID]} \
            --env BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS[$CI_CONFIG_ID]} \
            --env BUILD_RPC_INTERFACE=${BUILD_RPC_INTERFACE[$CI_CONFIG_ID]} \
            $DC_IMAGE_TAG"
                ;;

    delete-image )
        gcloud container images delete "$DC_IMAGE_TAG"
        ;;

    delete-vm )
        gcloud compute instances delete ${GCE_INSTANCE} --zone ${GCE_INSTANCE_ZONE}
        ;;

    ssh-vm )
        gcloud compute ssh ${GCE_INSTANCE} --zone ${GCE_INSTANCE_ZONE}
        ;;

    *)
        echo "Error: Unknown sub-command $1"
        ;;
esac
