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
import pickle
from collections import OrderedDict

import numpy as np
import open3d as o3d
import torch
import torch_cluster
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from benchmark_utils import measure_time, print_table_simple


OUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data")
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


# Define NNS methods
class Open3D:

    def __init__(self, device, search_type, index_type):
        self.device = device
        self.search_type = search_type
        self.index_type = {
            "int": o3d.core.Dtype.Int32,
            "long": o3d.core.Dtype.Int64
        }[index_type]

    def prepare_data(self, points, queries):
        points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        queries = o3d.core.Tensor(queries, dtype=o3d.core.Dtype.Float32)
        return points, queries

    def setup(self, points, queries, radius):
        points_dev = points.contiguous().to(self.device)
        queries_dev = queries.contiguous().to(self.device)
        index = o3d.core.nns.NearestNeighborSearch(points_dev, self.index_type)
        if self.search_type == "knn":
            index.knn_index()
        elif self.search_type == "radius":
            index.fixed_radius_index(radius)
        elif self.search_type == "hybrid":
            index.hybrid_index(radius)
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return {'index': index, 'queries': queries_dev}

    def search(self, index, queries, search_args):
        if self.search_type == "knn":
            out = index.knn_search(queries, search_args["k"])
        elif self.search_type == "radius":
            out = index.fixed_radius_search(queries, search_args["radius"])
        elif self.search_type == "hybrid":
            out = index.hybrid_search(queries, search_args["radius"],
                                      search_args["k"])
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return out

    def __str__(self):
        return f"{self.__class__.__name__}-{self.search_type.capitalize()}({self.device})"


class PyTorchCluster:

    def __init__(self, device, search_type):
        self.device = device
        self.search_type = search_type

    def prepare_data(self, points, queries):
        points = torch.from_numpy(points).float()
        queries = torch.from_numpy(queries).float()
        return points, queries

    def setup(self, points, queries, radius):
        points_dev = points.contiguous().to(self.device)
        queries_dev = queries.contiguous().to(self.device)
        return {'points': points_dev, 'queries': queries_dev}

    def search(self, points, queries, search_args):
        if self.search_type == "knn":
            out = torch_cluster.knn(points, queries, search_args["k"])
        elif self.search_type == "radius":
            out = torch_cluster.radius(points, queries, search_args["radius"], max_num_neighbors=100) # if # points > max, randomly pick the points.
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return out

    def __str__(self):
        return f"{self.__class__.__name__}-{self.search_type.capitalize()}({self.device})"


def compute_avg_radii(points, queries, neighbors):
    """Computes the radii based on the number of neighbors"""
    tree = cKDTree(points.numpy())
    avg_radii = []
    for k in neighbors:
        dist, _ = tree.query(queries.numpy(), k=k + 1)
        avg_radii.append(np.mean(dist.max(axis=-1)))
    return avg_radii


def prepare_benchmark_data(num_points, dimensions, num_queries=10):
    # setup dataset examples
    datasets = OrderedDict()

    # random dataset
    for D in dimensions:
        for num_points_ in num_points:
            N = int(num_points_)
            npy_file = os.path.join(OUT_DIR, f"random_D={D}_N={N}.npy")

            if not os.path.exists(npy_file):
                print(f"Generating a random dataset, random_D={D}_N={N}.npy...")
                points = np.random.randn(N, D)
                np.save(npy_file, points)

            print(f"Loading the random dataset, random_D={D}_N={N}.npy...")
            points = np.load(npy_file)
            queries = points.copy()[::num_queries]
            filename = os.path.basename(npy_file)
            datasets[filename] = {'points': points, 'queries': queries}

    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-st", "--search_type",
                        type=str,
                        default="knn",
                        choices=["knn", "radius", "hybrid"])
    parser.add_argument("-it", "--index_type",
                        type=str,
                        default="long",
                        choices=["int", "long"])
    parser.add_argument("-x", "--x_axis", type=str, default="num_points", choices=["num_points", "k"])
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    # devices
    device = o3d.core.Device(o3d.core.Device.CUDA, 0) if args.use_gpu else o3d.core.Device()

    # collects runtimes for all examples
    results = OrderedDict()
    datasets = prepare_benchmark_data(
        num_points=[1e2, 1e3, 1e4, 1e5, 1e6],
        dimensions=[3, 4, 8, 16, 32]
    )
    neighbors = [8, 16, 32, 64, 100]

    # prepare method
    methods = [
        Open3D(device, args.search_type, args.index_type),
        PyTorchCluster(torch.device('cuda' if args.use_gpu else 'cpu'), args.search_type),
    ]

    # run benchmark
    for method in methods:
        if not args.overwrite and os.path.exists(os.path.join(OUT_DIR, f"{method}.pkl")):
            print(f"Skip {method}...")
            continue

        for example_name, example in datasets.items():
            points, queries = example['points'], example['queries']
            if args.search_type == "knn":
                radii = neighbors
            else:
                radii = compute_avg_radii(points, queries, neighbors)
            print(f"\n{example_name} | {len(points)} points", end="")

            for k, radius in zip(neighbors, radii):
                points, queries = example['points'], example['queries']
                example_results = {'k': k, 'num_points': len(points)}
                search_args = {'k': k, 'radius': radius}

                if hasattr(method, "prepare_data"):
                    points, queries = method.prepare_data(points, queries)

                example_results["setup"] = measure_time(
                    lambda: method.setup(points, queries, radius)
                )
                setup_results = method.setup(points, queries, radius)

                if 'index' in setup_results.keys():
                    search_fn = lambda: method.search(setup_results['index'], setup_results['queries'], search_args)
                else:
                    search_fn = lambda: method.search(setup_results['points'], setup_results['queries'], search_args)

                time = measure_time(search_fn)
                example_results["search"] = time
                results[f"{example_name} n={len(points)} k={k}"] = example_results

                for v in setup_results.values():
                    del v
                o3d.core.cuda.release_cache()
                torch.cuda.empty_cache()

        # save results
        with open(os.path.join(OUT_DIR, f"{method}.pkl"), "wb") as f:
            pickle.dump(results, f)

    results = []
    for method in methods:
        with open(os.path.join(OUT_DIR, f"{method}.pkl"), "rb") as f:
            data = pickle.load(f)
            results.append(data)

    print_table_simple(methods, results)
