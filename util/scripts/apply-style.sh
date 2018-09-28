#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

set -e
set -u

num_files=0
num_files_formatted=0
script_dir=`cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd`

if ! [[ -x "$(command -v clang-format)" ]]; then
    echo "Error: clang-format is not installed."
    echo "On Ubuntu, run `apt-get install clang-format`"
    echo "On Mac, run `brew install clang-format`"
    exit 1
fi

pushd "${script_dir}/../.." > /dev/null
for dir in src examples ; do
    if ! [[ -d "${dir}" ]]; then
        echo "Directory $(pwd)/${dir} not found$"
        exit 1
    fi

    echo "Applying style in $(pwd)/${dir}"
    for src_file in $(find "${dir}" -type f -and \( -name '*.cpp' -or -name '*.h' \)); do
        file_diff=`diff -u <(cat ${src_file}) <(clang-format -style=file ${src_file}) || true`
        if [[ ! -z ${file_diff} ]]; then
            clang-format -style=file -i ${src_file}
            echo "[formatted] ${src_file}"
            num_files_formatted=$((num_files_formatted+1))
        fi
        num_files=$((num_files+1))
    done
done
popd > /dev/null

echo "${num_files_formatted} of ${num_files} files were reformatted"
