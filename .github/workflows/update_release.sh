#!/bin/bash
set -euo pipefail

for artifact in "$@"; do
    # extract filename supporting both POSIX and Windows-style paths
    # normalize backslashes to forward slashes for safe filename extraction
    norm_path="${artifact//\\/\/}"
    filename="${norm_path##*/}"  # same as $(basename "$norm_path")
    normalized_filename=$(echo "$filename" | sed -E \
        's/[0-9]+\.[0-9]+\.[0-9]+(\+[0-9a-f]{7,})?/<version>/g')

    echo "Uploading $filename (normalized: $normalized_filename)"
    # use normalized path for upload so single backslashes in Windows paths work
    gh -R isl-org/open3d release upload main-devel "$norm_path" --clobber

    for old_asset in $(gh -R isl-org/open3d release view main-devel --json assets --jq '.assets[] | .name' || echo ""); do
        normalized_old_asset=$(echo "$old_asset" | sed -E \
            's/[0-9]+\.[0-9]+\.[0-9]+(\+[0-9a-f]{7,})?/<version>/g')
        if [[ "$normalized_old_asset" == "$normalized_filename" &&
              "$old_asset" != "$filename" ]]; then
            echo "Deleting old asset: $old_asset"
            gh -R isl-org/open3d release delete-asset main-devel "$old_asset" -y || true
        fi
    done
done
gh -R isl-org/open3d release view main-devel
