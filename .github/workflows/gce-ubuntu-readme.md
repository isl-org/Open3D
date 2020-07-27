# Google compute engine setup for Open3D GPU CI


## Create new VM with disk
- Go to https://console.cloud.google.com/compute/instances
- Select the CI project (open3d-dev)
- Create new VM for every HW/OS combination. The current settings are:
    - VM instance name: `ci-gpu-ubuntu-instance-1`
    - Zone (+region): any. Current selection is `us-west1-b`.
    - HW: 4+ vCPU, 15+ GB RAM, 1+ GPU (Nvidia Tesla T4 or better).
    - OS: disk with OS Ubuntu 18.04 LTE (full).
- Note these variables in `gce-gpu-ubuntu.yml`
```yml
env:
  GCE_INSTANCE: ci-gpu-ubuntu-instance-1
  GCE_INSTANCE_ZONE: us-west1-b
```
and also set them in your local shell:
```bash
export GCE_INSTANCE=ci-gpu-ubuntu-instance-1  GCE_INSTANCE_ZONE=us-west1-b
```


## Install Google Cloud SDK and CLI tools
- On your local machine, follow these [instructions](https://cloud.google.com/sdk/install).
- For GitHub actions, use the action `GoogleCloudPlatform/github-actions/setup-gcloud`
- Alternately, the following steps can also be done from the Google cloud website through cloud ssh.

## Create service account and key

Reference: Steps 2.3-2.5 from `.travis/readme.md`

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

- Add key text to the [GitHub repository secrets](https://github.com/intel-isl/Open3D/settings/secrets) with name `GCE_SA_KEY`
- Also add secret `GCE_PROJECT: open3d-dev`


## Start VM instance
```bash
gcloud compute instances start ${GCE_INSTANCE} --zone=${GCE_INSTANCE_ZONE}
```
This may fail if res Google cloud does not have sufficient resources in the specific region. Try again in a few minutes.

## Clone source code repository

```bash
gcloud compute ssh ${GCE_INSTANCE} --zone ${GCE_INSTANCE_ZONE} -- -C \
           "git clone --recursive --depth 1 https://github.com/intel-isl/Open3D.git"
```

## Set up Python version

This installs Python3 and sets up symbolic links so that calls to `python` are handled by `python3`. Only the default version of Python for the OS (eg Python 3.6 for Ubuntu 18.04 and Python 3.8 for Ubuntu 20.04) is supported by this method.
```bash
gcloud compute ssh ${GCE_INSTANCE} --zone ${GCE_INSTANCE_ZONE} -- -C \
           "sudo apt-get update && \
           sudo apt-get --yes install python3 python3-pip && \
           echo  'Only for Ubuntu <= 20.04' && set -x && \
           ln -s /usr/bin/python3.6 \$HOME/.local/bin/python && \
           ln -s /usr/bin/python3.6-config \$HOME/.local/bin/python-config && \
           ln -s /usr/bin/pip3 \$HOME/.local/bin/pip"
```

## Shutdown VM instance

```bash
gcloud compute instances stop ${GCE_INSTANCE} --zone "$GCE_INSTANCE_ZONE"
```
