#!/bin/bash

if [[ $# < 1 ]]; then
    echo "Usage: $1 icon.svg"
    exit 1 
fi

svg=$1
filename=`basename "$svg"`
name=${filename%.*}
tmpdir="/tmp/${name}.iconset"
icnsfile="$name.icns"

rm -rf "$tmpdir"
mkdir -p "$tmpdir"

inkscape --export-png="$tmpdir/icon_512x512@2x.png" --export-width=1024 "$svg"
inkscape --export-png="$tmpdir/icon_512x512.png" --export-width=512 "$svg"
inkscape --export-png="$tmpdir/icon_256x256@2x.png" --export-width=512 "$svg"
inkscape --export-png="$tmpdir/icon_256x256.png" --export-width=256 "$svg"
inkscape --export-png="$tmpdir/icon_128x128@2x.png" --export-width=256 "$svg"
inkscape --export-png="$tmpdir/icon_128x128.png" --export-width=128 "$svg"
inkscape --export-png="$tmpdir/icon_32x32@2x.png" --export-width=64 "$svg"
inkscape --export-png="$tmpdir/icon_32x32.png" --export-width=32 "$svg"
inkscape --export-png="$tmpdir/icon_16x16@2x.png" --export-width=32 "$svg"
inkscape --export-png="$tmpdir/icon_16x16.png" --export-width=16 "$svg"

iconutil -c icns --output "$icnsfile" "$tmpdir"
rm -rf "$tmpdir"

# Inkscape is so chatty that we should output something
echo "Created $icnsfile"
