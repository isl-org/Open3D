#!/bin/bash
set -euo pipefail

for artifact in "$@"; do
    # extract filename supporting both POSIX and Windows-style paths
    # normalize backslashes to forward slashes for safe filename extraction
    norm_path="${artifact//\\/\/}"
    filename="${norm_path##*/}"  # same as $(basename "$norm_path")
    pattern=$(echo "$filename" | sed -E 's/([0-9]+\.[0-9]+\.[0-9]+\+)?[0-9a-f]{7,}/*/g')

    echo "Uploading $filename (pattern: $pattern)"
    # use normalized path for upload so single backslashes in Windows paths work
    gh release upload main-devel "$norm_path" --clobber

    for old_asset in $(gh release view main-devel --json assets --jq '.assets[] | .name' || echo ""); do
        # shellcheck disable=SC2254
        case "$old_asset" in
            $pattern)
                if [[ "$old_asset" != "$filename" ]]; then
                    echo "Deleting old asset: $old_asset"
                    gh release delete-asset main-devel "$old_asset" -y || true
                fi
                ;;
        esac
    done
done
gh release view main-devel
