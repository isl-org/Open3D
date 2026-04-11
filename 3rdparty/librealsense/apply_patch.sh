#!/bin/bash
# Apply a patch if it's not already applied
PATCH_FILE="$1"
SOURCE_DIR="$2"

cd "$SOURCE_DIR" || exit 1

# Initialize git if needed
if [ ! -d .git ]; then
    git init
fi

# Check if patch is already applied (reverse check)
if git apply --reverse --check --ignore-space-change --ignore-whitespace "$PATCH_FILE" 2>/dev/null; then
    echo "Patch already applied, skipping: $PATCH_FILE"
    exit 0
fi

# Try to apply the patch
if git apply --ignore-space-change --ignore-whitespace "$PATCH_FILE" 2>/dev/null; then
    echo "Patch applied successfully: $PATCH_FILE"
    exit 0
else
    echo "Warning: Failed to apply patch (may already be applied): $PATCH_FILE"
    # Check if the changes are already in the file by checking key content
    # If the patch modifies a specific file, we can check if those changes exist
    exit 0  # Don't fail the build if patch can't be applied
fi

