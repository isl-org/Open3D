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

import open3d as o3d
import numpy as np
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext, dirname, basename
import re
import sys
import struct
import zipfile
import os
import sys
if (sys.version_info > (3, 0)):
    pyver = 3
    from urllib.request import Request, urlopen
else:
    pyver = 2
    from urllib2 import Request, urlopen
import argparse


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


# get list of files inside a folder, matching the externsion, in sorted order.
def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


# converts kitti binary to pcd.
def bin_to_pcd(binFileName):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd


# preprocess and save in .ply format.
def preprocess_and_save(source_folder,
                        destination_folder,
                        voxel_size=0.05,
                        start_idx=0,
                        end_idx=1000):
    # get all files from the folder, and sort by name.
    filenames = get_file_list(source_folder, ".bin")

    print(
        "Converting .bin to .ply files and pre-processing from frame {} to index {}"
        .format(start_idx, end_idx))

    if (end_idx < start_idx):
        raise RuntimeError("End index must be smaller than start index.")
    if (end_idx > len(filenames)):
        end_idx = len(filenames)
        print(
            "WARNING: End index is greater than total file length, taking file length as end index."
        )

    filenames = filenames[start_idx:end_idx]
    for path in filenames:
        # convert kitti bin format to pcd format.
        pcd = bin_to_pcd(path)

        # downsample and estimate normals.
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
        voxel_down_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(),
            fast_normal_computation=False)

        # convert to Float32 dtype.
        tpcd = o3d.t.geometry.PointCloud.from_legacy(voxel_down_pcd)
        tpcd.point.positions = tpcd.point.positions.to(o3d.core.Dtype.Float32)
        tpcd.point.normals = tpcd.point.normals.to(o3d.core.Dtype.Float32)

        # extract name from path.
        name = str(path).rsplit('/', 1)[-1]
        name = name[:-3] + "ply"

        # write to the destination folder.
        output_path = destination_folder + name
        o3d.t.io.write_point_cloud(output_path, tpcd)


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


def get_kitti_sample_dataset(dataset_path, dataset_name):
    # data preparation
    download_parent_path = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    # download and unzip dataset
    path = join(dataset_path, dataset_name)
    if not os.path.exists(path):
        print("==================================")
        download_path = join(download_parent_path + dataset_name,
                             dataset_name + '_sync.zip')
        file_downloader(download_path)
        unzip_data("%s_sync.zip" % dataset_name,
                   "%s/%s" % (dataset_path, dataset_name))
        os.remove("%s_sync.zip" % dataset_name)
        print("")
    else:
        print(
            "The folder: {}, already exists. To re-download, kindly delete the folder and re-run this script."
            .format(path))


def find_source_pcd_folder_path(dataset_name):
    l = dataset_name.split('_')
    temp = l[0] + '_' + l[1] + '_' + l[2]
    dataset_name_parent = join(dataset_name, temp)
    dataset_name = join(dataset_name_parent,
                        dataset_name + '_sync/velodyne_points/data/')
    return dataset_name


valid_dataset_list = [
    "2011_09_26_drive_0009", "2011_09_30_drive_0018", "2011_09_30_drive_0027",
    "2011_09_30_drive_0028", "2011_10_03_drive_0027", "2011_10_03_drive_0034"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="2011_09_30_drive_0028",
        help='Kitti city sequence name [Example: "2011_09_26_drive_0009"].')
    parser.add_argument('--print_available_datasets',
                        action='store_true',
                        help='visualize ray casting every 100 frames')
    parser.add_argument('--voxel_size',
                        type=float,
                        default=0.05,
                        help='voxel size of the pointcloud.')
    parser.add_argument('--start_index',
                        type=int,
                        default=0,
                        help='start index of the dataset frame.')
    parser.add_argument('--end_index',
                        type=int,
                        default=1000,
                        help='maximum end index of the dataset frame.')

    args = parser.parse_args()

    if (args.print_available_datasets):
        for name in valid_dataset_list:
            print(name)
        sys.exit()

    if not args.dataset_name in valid_dataset_list:
        raise RuntimeError(
            "Dataset not present, kindly try with a different dataset. \nRun with --print_available_datasets, to get the list of available datasets."
        )

    download_dataset_path = "../../test_data/open3d_downloads/datasets/kitti_samples/"
    destination_path = join(download_dataset_path, "output/")

    # download and unzip dataset.
    get_kitti_sample_dataset(download_dataset_path, args.dataset_name)

    # get source path to raw dataset, and target path to processed dataset.
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    else:
        for f in os.listdir(destination_path):
            os.remove(os.path.join(destination_path, f))

    source_folder = join(download_dataset_path,
                         find_source_pcd_folder_path(args.dataset_name))
    print("Source raw kitti lidar data: ", source_folder)

    # convert bin to pcd, pre-process and save.
    preprocess_and_save(source_folder, destination_path, args.voxel_size,
                        args.start_index, args.end_index)

    print("Data fetching completed. Output pointcloud frames: ",
          destination_path)
