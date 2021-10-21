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

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # does not affect results

import argparse
import itertools
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

pwd = Path(os.path.dirname(os.path.realpath(__file__)))
open3d_root = pwd.parent.parent.parent

from benchmark_utils import measure_time, print_system_info, print_table


# Define NNS methods
class NNS:

    def __init__(self, device, search_type):
        self.device = device
        self.search_type = search_type

    def setup(self, points, queries, radius):
        points_dev = points.to(self.device)
        queries_dev = queries.to(self.device)
        index = o3d.core.nns.NearestNeighborSearch(points_dev)
        if self.search_type == "knn":
            index.knn_index()
        elif self.search_type == "radius":
            index.fixed_radius_index(radius)
        elif self.search_type == "hybrid":
            index.hybrid_index(radius)
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return index, queries_dev

    def search(self, index, queries, search_args):
        if self.search_type == "knn":
            index.knn_search(queries, search_args["knn"])
        elif self.search_type == "radius":
            index.fixed_radius_search(queries, search_args["radius"])
        elif self.search_type == "hybrid":
            index.hybrid_search(queries, search_args["radius"],
                                search_args["knn"])
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return ans

    def __str__(self):
        return f"{self.search_type.capitalize()}({self.device})"


def compute_avg_radii(points, queries, neighbors):
    """Computes the radii based on the number of neighbors"""
    tree = cKDTree(points.numpy())
    avg_radii = []
    for k in neighbors:
        dist, _ = tree.query(queries.numpy(), k=k + 1)
        avg_radii.append(np.mean(dist.max(axis=-1)))
    return avg_radii


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_type",
                        type=str,
                        default="knn",
                        choices=["knn", "radius", "hybrid"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # devices
    o3d_cpu_dev = o3d.core.Device()
    o3d_cuda_dev = o3d.core.Device(o3d.core.Device.CUDA, 0)

    # collects runtimes for all examples
    results = OrderedDict()
    # setup dataset examples
    datasets = OrderedDict()

    # TODO: remove hard-coded file list.
    files = [
        "small_tower.ply",
        "kitti_1.ply",
        "kitti_2.ply",
        "fluid_1000.ply",
        # "s3dis_1.ply", "s3dis_2.ply"
    ]
    for i, file in enumerate(files):
        filepath = os.path.join(open3d_root, file)
        pcd = o3d.t.io.read_point_cloud(filepath)
        points = queries = pcd.point['positions'].to(o3d.core.Dtype.Float32)
        queries = queries[::10]
        filename = os.path.basename(filepath)
        datasets[filename] = {'points': points, 'queries': queries}

    if args.search_type == "knn":
        # random data
        for dim in (3, 4, 8, 16, 32):
            points = o3d.core.Tensor.from_numpy(
                np.random.rand(100000, dim).astype(np.float32))
            queries = o3d.core.Tensor.from_numpy(
                np.random.rand(100000, dim).astype(np.float32))
            datasets['random{}'.format(dim)] = {
                'points': points,
                'queries': queries
            }

    # prepare methods
    methods = [
        NNS(o3d_cuda_dev, args.search_type),
        NNS(o3d_cpu_dev, args.search_type),
    ]
    neighbors = (1, 37, 64)

    # run benchmark
    for method in methods:
        if not args.overwrite and os.path.exists(f"{method}.pkl"):
            print(f"skip {method}")
            continue
        print(method)

        for example_name, example in datasets.items():
            points, queries = example['points'], example['queries']
            if args.search_type == "knn":
                radii = neighbors
            else:
                radii = compute_avg_radii(points, queries, neighbors)
            print(f"{example_name} {points.shape[0]}")

            for (knn, radius) in zip(neighbors, radii):
                points, queries = example['points'], example['queries']
                points = points.contiguous().to(o3d_cuda_dev)
                queries = queries.contiguous().to(o3d_cuda_dev)

                example_results = {'k': knn, 'num_points': points.shape[0]}

                if hasattr(method, "prepare_data"):
                    points, queries = method.prepare_data(points, queries)

                ans = measure_time(
                    lambda: method.setup(points, queries, radius))
                example_results['setup'] = ans

                index, queries = method.setup(points, queries, radius)

                ans = measure_time(lambda: method.search(
                    index, queries, dict(knn=knn, radius=radius)))
                example_results['search'] = ans

                results[
                    f'{example_name} n={points.shape[0]} k={knn}'] = example_results

                del index
                del points
                del queries
                o3d.core.cuda.release_cache()

        with open(f'{method}.pkl', 'wb') as f:
            pickle.dump(results, f)

    results = []
    for method in methods:
        with open(f"{method}.pkl", "rb") as f:
            print(f"{method}.pkl")
            data = pickle.load(f)
            results.append(data)

    print_system_info()
    print_table(methods, results)
