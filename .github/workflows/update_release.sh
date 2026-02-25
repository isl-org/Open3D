#!/bin/bash
set -euo pipefail

for artifact in "$@"; do
    filename=$(basename "$artifact")
    pattern=$(echo "$filename" | sed -E 's/([0-9]+\.[0-9]+\.[0-9]+\+)?[0-9a-f]{7,}/*/g')
    
    echo "Uploading $filename (pattern: $pattern)"
    gh release upload main-devel "$artifact" --clobber
    
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
