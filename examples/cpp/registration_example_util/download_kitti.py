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
def preprocess_and_save(source_folder, destination_folder, voxel_size=0.02):
    # get all files from the folder, and sort by name.
    filenames = get_file_list(source_folder, ".bin")

    print("Converting .bin to .ply files and pre-processing.")
    for path in filenames:
        # convert kitti bin format to pcd format.
        pcd = bin_to_pcd(path)

        # downsample and estimate normals.
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        voxel_down_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(),
            fast_normal_computation=False)

        # convert to Float32 dtype.
        tpcd = o3d.t.geometry.PointCloud.from_legacy_pointcloud(voxel_down_pcd)
        tpcd.point["points"] = tpcd.point["points"].to(o3d.core.Dtype.Float32)
        tpcd.point["normals"] = tpcd.point["normals"].to(o3d.core.Dtype.Float32)

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
            "The folder: %s, already exists. To re-download, kindly delete the folder and re-run this script."
            % path)


def find_source_pcd_folder_path(dataset_name):
    l = dataset_name.split('_')
    temp = l[0] + '_' + l[1] + '_' + l[2]
    dataset_name_parent = join(dataset_name, temp)
    dataset_name = join(dataset_name_parent,
                        dataset_name + '_sync/velodyne_points/data/')
    return dataset_name


valid_dataset_list = [
    "2011_09_26_drive_0001", "2011_09_26_drive_0002", "2011_09_26_drive_0005",
    "2011_09_26_drive_0009", "2011_09_26_drive_0011", "2011_09_26_drive_0013",
    "2011_09_26_drive_0014", "2011_09_26_drive_0015", "2011_09_26_drive_0017",
    "2011_09_26_drive_0018", "2011_09_26_drive_0019", "2011_09_26_drive_0020",
    "2011_09_26_drive_0022", "2011_09_26_drive_0023", "2011_09_26_drive_0027",
    "2011_09_26_drive_0028", "2011_09_26_drive_0029", "2011_09_26_drive_0032",
    "2011_09_26_drive_0035", "2011_09_26_drive_0036", "2011_09_26_drive_0039",
    "2011_09_26_drive_0046", "2011_09_26_drive_0048", "2011_09_26_drive_0051",
    "2011_09_26_drive_0052", "2011_09_26_drive_0056", "2011_09_26_drive_0057",
    "2011_09_26_drive_0059", "2011_09_26_drive_0060", "2011_09_26_drive_0061",
    "2011_09_26_drive_0064", "2011_09_26_drive_0070", "2011_09_26_drive_0079",
    "2011_09_26_drive_0084", "2011_09_26_drive_0086", "2011_09_26_drive_0087",
    "2011_09_26_drive_0091", "2011_09_26_drive_0093", "2011_09_26_drive_0095",
    "2011_09_26_drive_0096", "2011_09_26_drive_0101", "2011_09_26_drive_0104",
    "2011_09_26_drive_0106", "2011_09_26_drive_0113", "2011_09_26_drive_0117",
    "2011_09_26_drive_0119", "2011_09_28_drive_0001", "2011_09_28_drive_0002",
    "2011_09_28_drive_0016", "2011_09_28_drive_0021", "2011_09_28_drive_0034",
    "2011_09_28_drive_0035", "2011_09_28_drive_0037", "2011_09_28_drive_0038",
    "2011_09_28_drive_0039", "2011_09_28_drive_0043", "2011_09_28_drive_0045",
    "2011_09_28_drive_0047", "2011_09_28_drive_0053", "2011_09_28_drive_0054",
    "2011_09_28_drive_0057", "2011_09_28_drive_0065", "2011_09_28_drive_0066",
    "2011_09_28_drive_0068", "2011_09_28_drive_0070", "2011_09_28_drive_0071",
    "2011_09_28_drive_0075", "2011_09_28_drive_0077", "2011_09_28_drive_0078",
    "2011_09_28_drive_0080", "2011_09_28_drive_0082", "2011_09_28_drive_0086",
    "2011_09_28_drive_0087", "2011_09_28_drive_0089", "2011_09_28_drive_0090",
    "2011_09_28_drive_0094", "2011_09_28_drive_0095", "2011_09_28_drive_0096",
    "2011_09_28_drive_0098", "2011_09_28_drive_0100", "2011_09_28_drive_0102",
    "2011_09_28_drive_0103", "2011_09_28_drive_0104", "2011_09_28_drive_0106",
    "2011_09_28_drive_0108", "2011_09_28_drive_0110", "2011_09_28_drive_0113",
    "2011_09_28_drive_0117", "2011_09_28_drive_0119", "2011_09_28_drive_0121",
    "2011_09_28_drive_0122", "2011_09_28_drive_0125", "2011_09_28_drive_0126",
    "2011_09_28_drive_0128", "2011_09_28_drive_0132", "2011_09_28_drive_0134",
    "2011_09_28_drive_0135", "2011_09_28_drive_0136", "2011_09_28_drive_0138",
    "2011_09_28_drive_0141", "2011_09_28_drive_0143", "2011_09_28_drive_0145",
    "2011_09_28_drive_0146", "2011_09_28_drive_0149", "2011_09_28_drive_0153",
    "2011_09_28_drive_0154", "2011_09_28_drive_0155", "2011_09_28_drive_0156",
    "2011_09_28_drive_0160", "2011_09_28_drive_0161", "2011_09_28_drive_0162",
    "2011_09_28_drive_0165", "2011_09_28_drive_0166", "2011_09_28_drive_0167",
    "2011_09_28_drive_0168", "2011_09_28_drive_0171", "2011_09_28_drive_0174",
    "2011_09_28_drive_0177", "2011_09_28_drive_0179", "2011_09_28_drive_0183",
    "2011_09_28_drive_0184", "2011_09_28_drive_0185", "2011_09_28_drive_0186",
    "2011_09_28_drive_0187", "2011_09_28_drive_0191", "2011_09_28_drive_0192",
    "2011_09_28_drive_0195", "2011_09_28_drive_0198", "2011_09_28_drive_0199",
    "2011_09_28_drive_0201", "2011_09_28_drive_0204", "2011_09_28_drive_0205",
    "2011_09_28_drive_0208", "2011_09_28_drive_0209", "2011_09_28_drive_0214",
    "2011_09_28_drive_0216", "2011_09_28_drive_0220", "2011_09_28_drive_0222",
    "2011_09_28_drive_0225", "2011_09_29_drive_0004", "2011_09_29_drive_0026",
    "2011_09_29_drive_0071", "2011_09_29_drive_0108", "2011_09_30_drive_0016",
    "2011_09_30_drive_0018", "2011_09_30_drive_0020", "2011_09_30_drive_0027",
    "2011_09_30_drive_0028", "2011_09_30_drive_0033", "2011_09_30_drive_0034",
    "2011_09_30_drive_0072", "2011_10_03_drive_0027", "2011_10_03_drive_0034",
    "2011_10_03_drive_0042", "2011_10_03_drive_0047", "2011_10_03_drive_0058"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="2011_09_26_drive_0009",
        help='Kitti city sequence name [Example: "2011_09_26_drive_0009"].')
    parser.add_argument('--print_available_datasets',
                        action='store_true',
                        help='visualize ray casting every 100 frames')

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
    preprocess_and_save(source_folder, destination_path, 0.02)

    print("Data fetching completed. Output pointcloud frames: ",
          destination_path)
