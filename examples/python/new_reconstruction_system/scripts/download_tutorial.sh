#!/usr/bin/env bash
set -e

./gdrive_download.sh 1dkA6Tjh-aEie1J8qGlY1NlyJVP0_FpBX
unzip -qq tutorial.zip
mv tutorial ../dataset/
rm *.zip
