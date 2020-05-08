#!/usr/bin/env bash
set -e

# References:
# https://gist.github.com/mderazon/5a5b50d92f4c4adaf44d5f49dd95d0ef
# https://github.com/GoogleCloudPlatform/google-cloud-eclipse/blob/master/.travis.yml

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo "Running deploy_docs.sh..."

# Instal gsutil
if [ ! -d "$HOME/google-cloud-sdk/bin" ]; then
    rm -rf $HOME/google-cloud-sdk
    export CLOUDSDK_CORE_DISABLE_PROMPTS=1
    curl https://sdk.cloud.google.com | bash
fi
gcloud --quiet version
gcloud --quiet components update

if [ "$TRAVIS" = true ]; then
    # Decrypt key
    openssl aes-256-cbc \
        -K $encrypted_72034960ad12_key \
        -iv $encrypted_72034960ad12_iv \
        -in open3d-ci-sa-key.json.enc \
        -out open3d-ci-sa-key.json \
        -d

    # Autenticate with Google cloud
    gcloud auth activate-service-account --key-file open3d-ci-sa-key.json
    gcloud config set project isl-buckets
fi

# Compress the docs
commit_id="$(git rev-parse HEAD)"
docs_out_dir="${curr_dir}/../docs/_out" # Docs in ${docs_out_dir}/html
tar_file="${commit_id}.tar.gz"
rm -rf ${tar_file}
tar -C ${docs_out_dir} -czvf ${tar_file} html
gsutil cp ${tar_file} gs://open3d-docs/${tar_file}
