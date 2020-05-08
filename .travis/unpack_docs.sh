#!/usr/bin/env bash
set -e

# Find the lastest `master` commit in Open3D and unpack the corresponding docs
# from the Google Cloud bucket. This script is called by the documentation sever.

# Get master
master_commit=$(git ls-remote https://github.com/intel-isl/Open3D.git HEAD | awk '{ print $1}')
echo "master_commit: ${master_commit}"

# Download and untar docs
bucket_url="https://storage.googleapis.com/open3d-docs"
tar_url="${bucket_url}/${master_commit}.tar.gz"
echo "downloading url: ${tar_url}"
wget -c ${tar_url} -O - | tar -xz

# Deploy
source_dir=html
deployment_dir="/var/www/html/docs"
echo "source_dir: ${source_dir}"
ls ${source_dir}
echo "deployment_dir: ${deployment_dir}"

if [[ -f ${source_dir}/index.html && -f ${source_dir}/cpp_api/index.html ]]; then
    # Copy docs
    rm -rf ${deployment_dir}/latest
    cp -r ${source_dir} ${deployment_dir}/latest

    # Copy version tag
    rm -rf ${deployment_dir}/versions.js
    wget https://raw.githubusercontent.com/intel-isl/Open3D/master/docs/versions.js -P ${deployment_dir}
else
    echo "Python or C++ docs not found, keeping all old docs."
fi
