#!/usr/bin/env bash
set -euo pipefail

# __usage="USAGE:
#     $(basename $0) VM_NAME

#     Creates a VM with the given name. Upon successful VM creation, the zone
#     infomration will be written in /tmp/gcloud_vm_zone.txt.
# "

# OPEN3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. >/dev/null 2>&1 && pwd)"

# print_usage_and_exit() {
#     echo "$__usage"
#     exit 1
# }

# if [[ "$#" -ne 1 ]]; then
#     echo "Error: invalid number of arguments." >&2
#     print_usage_and_exit
# fi

# VM_NAME="$1"
# echo "Creating VM with name ${VM_NAME}"

# GCE_ZONES=(us-west1-a
#            us-west1-b
#            us-central1-a
#            us-central1-b
#            us-central1-f
#            us-east1-c
#            us-east1-d
#            us-east4-b
#            southamerica-east1-c
#            europe-west2-b
#            europe-west3-b
#            europe-west4-b
#            europe-west4-c
#            europe-west2-a
#            asia-southeast1-b
#            asia-southeast1-c
#            australia-southeast1-a)

gcloud compute instances create ${{ env.VM_NAME }} \
    --project open3d-dev \
    --service-account="${{ secrets.GCE_GPU_CI_SA }}" \
    --image-family common-cu110 \
    --image-project deeplearning-platform-release \
    --zone=australia-southeast1-a \
    --accelerator="count=2,type=nvidia-tesla-t4" \
    --maintenance-policy=TERMINATE \
    --machine-type=n1-standard-4 \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True,proxy-mode=project_editors"

# Wait for nvidia driver installation (~1min)
sleep 90s

gcloud compute ssh ${{ env.VM_NAME }} \
    --zone=australia-southeast1-a \
    --command "sudo gcloud auth configure-docker"

# GCE_ZID=0
# until ((GCE_ZID >= ${#GCE_ZONES[@]})) ||
#     gcloud compute instances create "$GCE_INSTANCE" \
#         --zone="${GCE_ZONES[$GCE_ZID]}" \
#         --accelerator="$GCE_GPU" \
#         --maintenance-policy=TERMINATE \
#         --machine-type=$GCE_INSTANCE_TYPE \
#         --boot-disk-size=$GCE_BOOT_DISK_SIZE \
#         --boot-disk-type=$GCE_BOOT_DISK_TYPE \
#         --image-family="$GCE_VM_CUSTOM_IMAGE_FAMILY" \
#         --service-account="$GCE_GPU_CI_SA"; do
#     ((GCE_ZID = GCE_ZID + 1))
# done
