#!/bin/bash
#

echo $#
if [[ $# != 6 ]]; then
    echo "Usage: $0 path/to/Open3d.app path/to/Open3d.entitlements apple-id cert_id team-id password"
    exit 1
fi

echo "Running as $1 $2 $3 $4 $5 $6"

# Sign app
echo "Signing $1 with entitlements $2 cert: $4..."
codesign --deep --force --options runtime --timestamp --entitlements $2 --sign $4 $1

# Verify signing worked
echo "Verifying signing..."
codesign -dvv --strict $1

appname=$1
zipname="${appname%.app}.zip"

# Create zip with ditto
echo "Zipping to prepare for notarization..."
ditto -c -k --rsrc --keepParent $1 $zipname

# Send signed app in for notarization
# Note: this command returns the result
echo "Submitting for notarization..."
xcrun notarytool submit $zipname --apple-id $3 --team-id $5 --password $6 --wait

# Staple the original app
xcrun stapler staple $1

# Delete old zip and create a new one for distribution
rm $zipname
ditto -c -k --keepParent $1 $zipname
