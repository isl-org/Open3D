import math
import os
import sys
import shutil
import argparse
sys.path.append("../../Utility")
from file import *
# original code is written by Andrew. W. Chen
# input: openni style unsynchronized color and depth images
# output: synchronized color and depth images


def run_synchronization(args):
    folder_path = args.dataset
    color_files, depth_files = get_rgbd_file_lists(folder_path)
    if args.debug_mode:
        print(depth_files)
        print(color_files)

    # filename format is:
    # frame-timestamp.filetype
    timestamps = {
        'depth': [None] * len(depth_files),
        'color': [None] * len(color_files)
    }
    for i, name in enumerate(depth_files):
        depth_timestamp = int(
            os.path.basename(depth_files[i]).replace('-', '.').split('.')[1])
        timestamps['depth'][i] = depth_timestamp
    for i, name in enumerate(color_files):
        color_timestamp = int(
            os.path.basename(color_files[i]).replace('-', '.').split('.')[1])
        timestamps['color'][i] = color_timestamp

    # associations' index is the color frame, and the value at
    # that index is the best depth frame for the color frame
    associations = []
    depth_idx = 0
    for i in range(len(color_files)):
        best_dist = float('inf')
        while depth_idx <= len(depth_files) - 1 and i <= len(color_files) - 1:
            dist = math.fabs(timestamps['depth'][depth_idx] - \
                    timestamps['color'][i])
            if dist > best_dist:
                break
            best_dist = dist
            depth_idx += 1
            if depth_idx > timestamps['depth'][-1]:
                print("Ended at color frame %d, depth frame %d" %
                      (i, depth_idx))
        associations.append(depth_idx - 1)
        if args.debug_mode:
            print("%d %d %d %d" %
                  (i, depth_idx - 1, timestamps['depth'][depth_idx - 1],
                   timestamps['color'][i]))

    os.rename(os.path.join(folder_path, "depth"),
              os.path.join(folder_path, "temp"))
    if not os.path.exists(os.path.join(folder_path, "depth")):
        os.makedirs(os.path.join(folder_path, "depth"))
    for i, assn in enumerate(associations):
        temp_name = os.path.join(folder_path, "temp",
                                 os.path.basename(depth_files[assn]))
        new_name = os.path.join(folder_path, "depth/%06d.png" % (i + 1))
        if args.debug_mode:
            print(temp_name)
            print(new_name)
        if not exists(temp_name):
            assert (i + 1 == len(color_files))
            os.remove(color_files[-1])
        else:
            os.rename(temp_name, new_name)
    shutil.rmtree(os.path.join(folder_path, "temp"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Synchronize color and depth images")
    parser.add_argument("dataset", help="path to the dataset")
    parser.add_argument("--debug_mode",
                        help="turn on debug mode",
                        action="store_true")
    args = parser.parse_args()
    run_synchronization(args)
