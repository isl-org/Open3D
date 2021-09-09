#!/usr/bin/env bash
set -euo pipefail

# The following environment variables are assumed
echo "VM_NAME: ${VM_NAME}"
echo "GCE_PROJECT: ${GCE_PROJECT}"     # Hidden by GHA
echo "GCE_GPU_CI_SA: ${GCE_GPU_CI_SA}" # Hidden by GHA

GCE_ZONES=(
    us-west1-a
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
    australia-southeast1-a
)

GCE_ZID=0
until ((GCE_ZID >= ${#GCE_ZONES[@]})) ||
    gcloud compute instances create ${VM_NAME} \
        --project=${GCE_PROJECT} \
        --zone="${GCE_ZONES[$GCE_ZID]}" \
        --service-account="${GCE_GPU_CI_SA}" \
        --image-family common-cu110 \
        --image-project deeplearning-platform-release \
        --accelerator="count=2,type=nvidia-tesla-t4" \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-4 \
        --boot-disk-type=pd-ssd \
        --metadata="install-nvidia-driver=True,proxy-mode=project_editors"; do
    ((GCE_ZID = GCE_ZID + 1))
done

# Export environment variable for next step
export VM_ZONE
VM_ZONE="${GCE_ZONES[$GCE_ZID]}"
echo "VM created in VM_ZONE=${VM_ZONE}"
echo "VM_ZONE=${VM_ZONE}" >> "$GITHUB_ENV"

# Wait for nvidia driver installation (~1min)
sleep 90s

# 0 => success
exit $((GCE_ZID >= ${#VM_ZONE[@]}))
