# Running workflows on GitHub Actions

## Documentation deployment

### Directory structure

- `.github/workflows/documentation.yml`: Github Actions workflow file to
  create and deploy documentation. Documentation is created for every branch
  as a CI test, but deployed only for `master`.
- `util/ci_utils.sh:build_docs()`: Called by GitHub Actions to build documentation.
- `unpack_docs.sh`: Called by the documentation server to deploy the docs into
  the website.

### Setting up the docs deployment

#### Step 1: Google Cloud

1. Setup `gcloud`
    - [Install `gcloud`](https://cloud.google.com/sdk/install)
    - `gcloud init` and login with admin's google account on the web
2. Create storage bucket
    ```bash
    # Create with uniform bucket-level access: `-b`
    # https://cloud.google.com/storage/docs/creating-buckets#storage-create-bucket-gsutil
    # gsutil mb -p [PROJECT_ID] -c [STORAGE_CLASS] -l [BUCKET_LOCATION] -b on gs://[BUCKET_NAME]/
    gsutil mb -p isl-buckets -c STANDARD -l US -b on gs://open3d-docs/

    # Grant public read permission
    # The current user must have appropriate permission, you may do this in the web interface
    gsutil acl ch -u AllUsers:R gs://open3d-docs/

    # Set object life cycle
    # https://cloud.google.com/storage/docs/managing-lifecycles#delete_an_object
    gsutil lifecycle set gcs.lifecycle.json gs://open3d-docs/
    ```

    Objects will be stored in the bucket for one week. Currently, the
    documentation server fetches the latest docs from `master` branch every hour.
    If the documentation server fails to fetch the docs matching the `master`
    commit id, the last successfully fetched docs will be displayed.
3. Create service account
    ```bash
    gcloud iam service-accounts create open3d-ci-sa \
        --description="Service account for Open3D CI" \
        --display-name="open3d-ci-sa"
    ```
4. Grant `objectAdmin` to the service account
    ```bash
    gsutil iam ch \
        serviceAccount:open3d-ci-sa@isl-buckets.iam.gserviceaccount.com:objectAdmin \
        gs://open3d-docs
    ```
5. Create key for service account
    ```bash
    gcloud iam service-accounts keys create ~/open3d-ci-sa-key.json \
    --iam-account open3d-ci-sa@isl-buckets.iam.gserviceaccount.com
    ```

Now `~/open3d-ci-sa-key.json` should have been created.

#### Step 2: GitHub

1. Encode the private key json file with `base64 ~/open3d-ci-sa-key.json` and
    add the output text to the
    [GitHub repository secrets](https://github.com/isl-org/Open3D/settings/secrets)
    with name `GCE_SA_KEY_DOCS_CI`

2. Also add secret `GCE_DOCS_PROJECT: isl-buckets`

## Google compute engine setup for GPU CI

### CI Procedure

The GCE CI workflow `.github/workflows/gce-ubuntu-docker.yml` performs these steps:

- Clone the repository
- Build docker image, starting with a an NVIDIA base devel image with CUDA and
  cuDNN.
- Push image to Google container registry.
- On Google Compute Engine (GCE), in parallel (up to GPU quota limit - currently
  4):
- Create a new VM instance with a custom OS image
- Run docker image on GCE (Google Compute Engine) with environment variables
  set for specific build config.
- The docker image entrypoint is the `run-ci.sh` script: build, install, run
  tests and uninstall.
- Delete the VM instance.

A separate VM instance is created for each commit and build option. The VM
instances are named according to the commit hash and build config ID used. We
cycle through 13 different US GCE zones if VM creation fails in the first zone,
either due to lack of resources or GPU quota exhaustion.

### Setup

#### Step 1: Google Cloud: Create service account and key

1. Create service account
    ```bash
    gcloud iam service-accounts create open3d-ci-sa \
        --description="Service account for Open3D CI" \
        --display-name="open3d-ci-sa"    \
        --project open3d-dev
    ```
2. Grant `Compute Instance Admin (beta)` to the service account
    ```bash
    gcloud projects add-iam-policy-binding open3d-dev \
        --member=serviceAccount:open3d-ci-sa-gpu@open3d-dev.iam.gserviceaccount.com \
        --role=roles/compute.instanceAdmin \
        --project open3d-dev
    ```
3. Create key for service account
    ```bash
    gcloud iam service-accounts keys create ~/open3d-ci-sa-key.json \
        --iam-account open3d-ci-sa@open3d-dev.iam.gserviceaccount.com \
        --project open3d-dev
    ```
    Now `~/open3d-ci-sa-key.json` should have been created.

#### Step 2: Google Cloud: Create custom VM image

```bash
./util/docker/open3d-gpu/scripts/gce-ubuntu-docker-run.sh create-base-vm-image
```

The custom VM image has NVIDIA drivers, `nvidia-container-toolkit` and `docker`
installed. It contains today's date in the name and the image family is set to
`ubuntu-os-docker-gpu-2004-lts`. The latest image from this family is
used for running CI.

#### Step 3: GitHub

1.  Encode the private key json file with `base64 ~/open3d-ci-sa-key.json` and
    add the output text to the
    [GitHub repository secrets](https://github.com/isl-org/Open3D/settings/secrets)
    with name `GCE_SA_KEY_GPU_CI`
2.  Also add secret `GCE_PROJECT: open3d-dev`

## Ccache strategy

- Typically, a build generates ~500MB cache. A build with Filament compiled from
  source generates ~600MB cache.
- Typically, regular X86 Ubuntu and macOS builds take about 40 mins without
  caching.
- The bottleneck of the CI is in the ARM build since it runs on a simulator.
  When building Filament from source, the build time can exceed GitHub's 6-hour
  limit if caching is not properly activated. With proper caching and good cache
  hit rate, the ARM build job can run within 1 hour.
- Both Ubuntu and macOS have a max cache setting of 2GB each out of a total
  cache limit of 5GB for the whole repository. Windows MSVC does not use
  caching at present since `ccache` does not officially support MSVC.
- ARM64 cache (limit 1.5GB) is stored on Google cloud bucket
  (`open3d-ci-cache` in the `isl-buckets` project). The bucket is world
  readable, but needs the `open3d-ci-sa` service account for writing. Every
  ARM64 build downloads the cache contents before build. Only `master` branch
  builds use `gsutil rsync` to update the cache in GCS. Cache transfer only
  takes a few minutes, but reduces ARM64 CI time to about 1:15 hours.

## Development wheels for user testing

`master` branch Python wheels are uploaded to a world readable GCS bucket for
users to try out development wheels.

### Google Cloud storage

Follow instructions in A. Documentation deployment to setup a Google cloud
bucket with:

- Project: open3d-dev
- Service account: open3d-ci-sa-gpu
- Bucket name: open3d-ci-sa-gpu
- Public read permissions
- One week object lifecycle

```bash
gsutil mb -p open3d-dev -c STANDARD -l US -b on gs://open3d-releases-master
gsutil acl ch -u AllUsers:R gs://open3d-releases-master
gsutil lifecycle set gcs.lifecycle.json gs:/open3d-releases-master
gsutil iam ch \
    serviceAccount:open3d-ci-sa-gpu@open3d-dev.iam.gserviceaccount.com:objectAdmin \
    gs://open3d-releases-master
```
