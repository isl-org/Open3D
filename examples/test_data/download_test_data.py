# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
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

# Download Open3D test data files. The default download path is
# Open3D/examples/test_data/open3d_downloads
#
# See https://github.com/intel-isl/open3d_downloads for details on how to
# manage the test data files.
#
# We have to put the version check here and the rest of the Python 3.6+
# compatible code in a separate file. Otherwise, Python 2 complains about syntax
# errors before the version check. In addition, please keep this file simple and
# Python 2&3 compatible. See https://stackoverflow.com/a/3760194/1255535.
import sys
if sys.version_info < (3, 6):
    raise RuntimeError(
        "Python version must be >= 3.6, however, Python {}.{} is used.".format(
            sys.version_info[0], sys.version_info[1]))

import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_utils import download_all_files

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Open3d test data.")
    parser.add_argument("--overwrite",
                        action="store_true",
                        default=False,
                        help="Overwrite existing file.")
    args = parser.parse_args()
    download_all_files(overwrite=args.overwrite)
