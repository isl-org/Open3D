#!/usr/bin/env bash

## Shell scripts for GCE CI workflow, called from the github actions yml file
## (for portability and ease of maintenance)

## These variables should be in the environment (from github repo secrets)
    # GCE_PROJECT
    # GCE_GPU_CI_SA
## GITHUB environment
    # GITHUB_SHA
## CI test matrix:
    # SHARED
    # BUILD_DEPENDENCY_FROM_SOURCE
    # BUILD_TENSORFLOW_OPS
    # BUILD_PYTORCH_OPS
    # BUILD_CUDA_MODULE

NPROC=8                      # {2,4,8,16,32,64,96}
GCE_INSTANCE_BASENAME=ci-gpu-vm
GCE_INSTANCE_BASE_TYPE=n1-standard      # GCE only allows n1-standard machines with GPUs (2020/07)
GCE_INSTANCE_ZONE=us-west1-b
GCE_GPU=count=1,type=nvidia-tesla-t4
GCE_BOOT_DISK_TYPE=pd-ssd
GCE_BOOT_DISK_SIZE=32GB
NVIDIA_DRIVER_VERSION=440    # Must be present in Ubuntu repos
GCE_IMAGE_FAMILY=ubuntu-1804-lts
IMAGE=open3d-gpu-ci-base
REGISTRY_HOSTNAME=gcr.io


case "$1" in
    gcloud-setup )
        gcloud auth configure-docker
        gcloud info
        ;;

    docker-build )
        docker build -t "$REGISTRY_HOSTNAME"/"$IMAGE":"$GITHUB_SHA" \
            --build-arg GITHUB_SHA="$GITHUB_SHA" \
        ;;

    docker-push )
          docker push "$REGISTRY_HOSTNAME/$GCE_PROJECT/$IMAGE":"$GITHUB_SHA"
        ;;

    create-vm )
          GCE_INSTANCE_TYPE=${GCE_INSTANCE_BASE_TYPE}-$NPROC
          GCE_INSTANCE=${GCE_INSTANCE_BASENAME}-${GITHUB_SHA::8}
          TRIES=0       # Try creating a VM instance 5 times
          while (( TRIES<5 )) && ! \
          # Create VM and deploy container, but only run no-op for now
          gcloud compute instances create-with-container $GCE_INSTANCE \
            --zone=${GCE_INSTANCE_ZONE} \
            --accelerator=$GCE_GPU \
            --maintenance-policy=TERMINATE
            --machine-type=$GCE_INSTANCE_TYPE \
            --boot-disk-size=$GCE_BOOT_DISK_SIZE \
            --boot-disk-type=$GCE_BOOT_DISK_TYPE \
            --metadata=startup-script="/bin/bash -c cos-extensions install gpu" \
            --service-account=$GCE_GPU_CI_SA \
            --container-image="$REGISTRY_HOSTNAME/$GCE_PROJECT/$IMAGE:$GITHUB_SHA" \
            --container-command=true \
            --container-restart-policy=never
          do
            sleep 5
            (( TRIES=TRIES+1 ))
          done
          exit $(( TRIES>=5 ))  # 0 => success
        ;;


    run-ci )
        gcloud compute ssh ${GCE_INSTANCE} --zone ${GCE_INSTANCE_ZONE} -- -C \
            "docker run --rm --gpus all \
            --env SHARED=$SHARED \
            --env NPROC=$NPROC \
            --env BUILD_DEPENDENCY_FROM_SOURCE=$BUILD_DEPENDENCY_FROM_SOURCE \
            --env BUILD_CUDA_MODULE=$BUILD_CUDA_MODULE \
            --env BUILD_TENSORFLOW_OPS=$BUILD_TENSORFLOW_OPS \
            --env BUILD_PYTORCH_OPS=$BUILD_PYTORCH_OPS \
            --env LOW_MEM_USAGE=$LOW_MEM_USAGE \
            $REGISTRY_HOSTNAME/$GCE_PROJECT/$IMAGE:$GITHUB_SHA"
                    ;;

    cleanup )
        gcloud compute instances delete ${GCE_INSTANCE} --zone ${GCE_INSTANCE_ZONE}
        gcloud container images delete "$REGISTRY_HOSTNAME/$GCE_PROJECT/$IMAGE:$GITHUB_SHA"
