# Travis and deployment

## Directory structure

- `deploy_docs.sh`: Called by Travis on `master` branch. This will copy the docs
  html files in the `docs` folder, compress them and upload to Google Cloud
  bucket.
- `unpack_docs.sh`: Called by the documentation server to deploy the docs into
  the website.

## Notes on setting up the docs deployment

### Step 1: Travis CLI

1. Install Travis client

   ```bash
   sudo apt install ruby-dev
   sudo gem install travis
   travis login --github-token xxxxxxxxxx # Use your GitHub access token
   ```

### Step 2: Google Cloud

1. Setup `gcloud`

   - Install `gcloud`
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

### Step 3: Encrypt key file

1. Encrypt file using Travis

   ```bash
   # Copy the key file to the Open3D git repo
   cd /your/path/to/repo/Open3D/.travis
   cp ~/open3d-ci-sa-key.json .  # Never add this file to git!
   travis encrypt-file open3d-ci-sa-key.json
   ```

2. Copy the decryption script to `.travis.yml`, modify path if necessary, e.g.

   ```bash
   openssl aes-256-cbc -K $encrypted_72034960ad12_key -iv $encrypted_72034960ad12_iv -in open3d-ci-sa-key.json.enc -out open3d-ci-sa-key.json -d
   ```

3. Commit the change

   ```bash
   cd /your/path/to/repo/Open3D
   git add .travis/open3d-ci-sa-key.json.enc
   git commit -m "Add travis key"
   ```
