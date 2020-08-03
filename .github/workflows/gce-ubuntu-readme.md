# Google compute engine setup for Open3D GPU CI

The GCE CI workflow performs these steps:
- Creates a new VM instance with a blank OS image
- Clones the repository and checks out the specific commit
- Installs dependencies (including NVIDIA drivers and CUDA)
- Builds the code and runs tests (`run-ci.sh`)
- Deletes the VM instance.

A separate VM instance is created for each commit and build option. The VM instances are named according to the commit hash and build options used.

## Current VM settings
```yaml
  - NPROC: 8                      # {2,4,8,16,32,64,96}
  - GCE_PROJECT: open3d-dev
  - GCE_INSTANCE_BASENAME: ci-gpu-vm
  - GCE_INSTANCE_BASE_TYPE: n1-standard
  - GCE_INSTANCE_ZONE: us-west1-b
  - GCE_IMAGE_FAMILY: ubuntu-1804-lts
  - GCE_GPU: count=1,type=nvidia-tesla-t4
  - GCE_BOOT_DISK_TYPE: pd-ssd
  - NVIDIA_DRIVER_VERSION: 440    # Must be present in Ubuntu repos
```

## Create service account and key

Reference: Adapted from `.travis/readme.md`

1. Setup `gcloud`

   - Install `gcloud` CLI
   - `gcloud init` and login with admin's google account on the web

3. Create service account
   ```bash
   gcloud iam service-accounts create open3d-ci-sa \
       --description="Service account for Open3D CI" \
       --display-name="open3d-ci-sa"    \
       --project open3d-dev
   ```

4. Grant `Compute Instance Admin (beta)` to the service account
   ```bash
   gcloud projects add-iam-policy-binding open3d-dev \
       --member=serviceAccount:open3d-ci-sa-gpu@open3d-dev.iam.gserviceaccount.com \
       --role=roles/compute.instanceAdmin \
       --project open3d-dev
   ```

5. Create key for service account
   ```bash
   gcloud iam service-accounts keys create ~/open3d-ci-sa-key.json \
       --iam-account open3d-ci-sa@open3d-dev.iam.gserviceaccount.com \
       --project open3d-dev
   ```
   Now `~/open3d-ci-sa-key.json` should have been created.

- Encode the private key json file with `base64 - < ~/open3d-ci-sa-key.json` and add the output text to the [GitHub repository secrets](https://github.com/intel-isl/Open3D/settings/secrets) with name `GCE_SA_KEY_GPU_CI`
- Also add secret `GCE_PROJECT: open3d-dev`
