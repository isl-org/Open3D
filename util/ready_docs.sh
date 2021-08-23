#!/usr/bin/env bash
set -eu

if [ -z "${docs_tar_file-}" ]; then
    echo "docs_tar_file is unset"
    exit 1
else
    echo "docs_tar_file is set to ${docs_tar_file}"
fi

remote_src=gs://open3d-docs/wait_for_wheels_${docs_tar_file}
remote_dst=gs://open3d-docs/${docs_tar_file}

# Retry for 3 hours (2 * 90 = 180 minutes).
total_retry=90
retry_gap=2
n=0
until [ "${n}" -ge "${total_retry}" ]
do
   gsutil mv "${remote_src}" "${remote_dst}" && break
   n=$((n+1))
   echo "Attempt (${n}/${total_retry}), retrying in ${retry_gap} minutes."
   sleep ${retry_gap}m
done

if gsutil -q stat "${remote_dst}"; then
    echo "All development wheels available. Documentation archive is ready for deployment:"
    echo "https://storage.googleapis.com/open3d-docs/${docs_tar_file}"
else
    echo "Failed to rename the documentation archive."
    exit 1
fi
