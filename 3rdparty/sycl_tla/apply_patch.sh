#!/bin/bash
# Apply the oneAPI compatibility patch once per sycl-tla checkout.
set -e

PATCH_FILE="$1"
SOURCE_DIR="$2"

cd "$SOURCE_DIR"
if git apply --reverse --check --ignore-space-change --ignore-whitespace \
        "$PATCH_FILE" 2>/dev/null; then
    echo "sycl-tla compatibility patch already applied"
else
    git apply --ignore-space-change --ignore-whitespace "$PATCH_FILE"
fi
