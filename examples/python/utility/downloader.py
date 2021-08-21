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

# examples/python/utility/downloader.py

import zipfile
import os
import sys
if (sys.version_info > (3, 0)):
    pyver = 3
    from urllib.request import Request, urlopen
else:
    pyver = 2
    from urllib2 import Request, urlopen

# dataset from redwood-data.org
dataset_names = ["livingroom1", "livingroom2", "office1", "office2"]
dataset_path = "testdata/"


def get_redwood_dataset():
    # data preparation
    if not os.path.exists(dataset_path):
        # download and unzip dataset
        for name in dataset_names:
            print("==================================")
            file_downloader("https://github.com/isl-org/open3d_downloads/releases/download/redwood/%s-fragments-ply.zip" % \
                    name)
            unzip_data("%s-fragments-ply.zip" % name,
                       "%s/%s" % (dataset_path, name))
            os.remove("%s-fragments-ply.zip" % name)
            print("")


def file_downloader(url):
    file_name = url.split('/')[-1]
    u = urlopen(url)
    f = open(file_name, "wb")
    if pyver == 2:
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
    elif pyver == 3:
        file_size = int(u.getheader("Content-Length"))
    print("Downloading: %s " % file_name)

    file_size_dl = 0
    block_sz = 8192
    progress = 0
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        if progress + 10 <= (file_size_dl * 100. / file_size):
            progress = progress + 10
            print(" %.1f / %.1f MB (%.0f %%)" % \
                    (file_size_dl/(1024*1024), file_size/(1024*1024), progress))
    f.close()


def unzip_data(path_zip, path_extract_to):
    print("Unzipping %s" % path_zip)
    zip_ref = zipfile.ZipFile(path_zip, 'r')
    zip_ref.extractall(path_extract_to)
    zip_ref.close()
    print("Extracted to %s" % path_extract_to)
