# Google compute engine setup for Open3D GPU CI

The GCE CI workflow performs these steps:

-   Clone the repository

-   Build docker image, starting with a an NVIDIA base devel image with CUDA and
    cuDNN.

-   Push image to Google container registry.

-   On Google Compute Engine (GCE), in parallel (up to GPU quota limit - currently
    4):
    -   Create a new VM instance with a custom OS image

    -   Run docker image on GCE (Google Compute Engine) with environment variables
        set for specific build config.

    -   The docker image entrypoint is the `run-ci.sh` script: build, install, run
        tests and uninstall.

    -   Delete the VM instance.

A separate VM instance is created for each commit and build option. The VM
instances are named according to the commit hash and build config ID used. We
cycle through 13 different US GCE zones if VM creation fails in the first zone,
either due to lack of resources or GPU quota exhaustion.

## Custom VM image creation

```bash
./util/docker/open3d-gpu/scripts/gce-ubuntu-docker-run.sh create-base-vm-image
```

The custom VM image has NVIDIA drivers, `nvidia-container-toolkit` and `docker`
installed. It contains today's date in the name and the image family is set to
`ubuntu-os-docker-gpu-2004-lts`. The latest image from this family is
used for running CI.

## Create service account and key

Reference: Adapted from `.travis/readme.md`

1.  Setup `gcloud`

    -   Install `gcloud` CLI
    -   `gcloud init` and login with admin's google account on the web

2.  Create service account

    ```bash
    gcloud iam service-accounts create open3d-ci-sa \
        --description="Service account for Open3D CI" \
        --display-name="open3d-ci-sa"    \
        --project open3d-dev
    ```

3.  Grant `Compute Instance Admin (beta)` to the service account

    ```bash
    gcloud projects add-iam-policy-binding open3d-dev \
        --member=serviceAccount:open3d-ci-sa-gpu@open3d-dev.iam.gserviceaccount.com \
        --role=roles/compute.instanceAdmin \
        --project open3d-dev
    ```

4.  Create key for service account

    ```bash
    gcloud iam service-accounts keys create ~/open3d-ci-sa-key.json \
        --iam-account open3d-ci-sa@open3d-dev.iam.gserviceaccount.com \
        --project open3d-dev
    ```

    Now `~/open3d-ci-sa-key.json` should have been created.

-   Encode the private key json file with `base64 - < ~/open3d-ci-sa-key.json` and
    add the output text to the
    [GitHub repository secrets](https://github.com/intel-isl/Open3D/settings/secrets)
    with name `GCE_SA_KEY_GPU_CI`

-   Also add secret `GCE_PROJECT: open3d-dev`
