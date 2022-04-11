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
from scipy.spatial import cKDTree
import nvidia_smi
import matplotlib.pyplot as plt

from benchmark_utils import measure_memory, print_system_info, print_table_memory


# Define NNS methods
class NNS:

    def __init__(self, device, search_type, index_type):
        assert index_type in ["int", "long"]
        self.device = device
        self.search_type = search_type
        self.index_type = o3d.core.Int32 if index_type == "int" else o3d.core.Int64

    def setup(self, points, queries, radius):
        points_dev = points.to(self.device)
        queries_dev = queries.to(self.device)
        index = o3d.core.nns.NearestNeighborSearch(points_dev, self.index_type)
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
            out = index.knn_search(queries, search_args["knn"])
        elif self.search_type == "radius":
            out = index.fixed_radius_search(queries, search_args["radius"])
        elif self.search_type == "hybrid":
            out = index.hybrid_search(queries, search_args["radius"],
                                      search_args["knn"])
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return out

    def __str__(self):
        return f"{self.search_type.capitalize()}({self.device}, {self.index_type})_memory"


def compute_avg_radii(points, queries, neighbors):
    """Computes the radii based on the number of neighbors"""
    tree = cKDTree(points.numpy())
    avg_radii = []
    for k in neighbors:
        dist, _ = tree.query(queries.numpy(), k=k + 1)
        avg_radii.append(np.mean(dist.max(axis=-1)))
    return avg_radii


def prepare_benchmark_data():
    # setup dataset examples
    datasets = OrderedDict()

    # random dataset
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "testdata")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    log10_ns = [4, 5, 6, 7]

    for log10_n in log10_ns:
        print("==================================")
        npy_file = os.path.join(out_dir, f"random_1e{log10_n}.npy")

        if not os.path.exists(npy_file):
            print(f"generating a random dataset, random_1e{log10_n}.npy...")
            N = int(np.power(10, log10_n))
            points = np.random.randn(N, 3)
            np.save(npy_file, points)

        print(f"loading the random dataset, random_1e{log10_n}.npy...")
        points = queries = o3d.core.Tensor(np.load(npy_file),
                                           dtype=o3d.core.Float32)
        queries = queries[::10]
        filename = os.path.basename(npy_file)
        datasets[filename] = {'points': points, 'queries': queries}
        print("")
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_type",
                        type=str,
                        default="knn",
                        choices=["knn", "radius", "hybrid", "all"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu_idx", type=int, default=3)
    args = parser.parse_args()

    # devices
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(args.gpu_idx)
    o3d_cpu_dev = o3d.core.Device()
    o3d_cuda_dev = o3d.core.Device(o3d.core.Device.CUDA, 0)

    # collects runtimes for all examples
    results = OrderedDict()

    datasets = prepare_benchmark_data()

    # prepare methods
    if args.search_type == "all":
        methods = [
            NNS(o3d_cuda_dev, "knn", "int"),
            NNS(o3d_cuda_dev, "knn", "long"),
            NNS(o3d_cuda_dev, "radius", "int"),
            NNS(o3d_cuda_dev, "radius", "long"),
            NNS(o3d_cuda_dev, "hybrid", "int"),
            NNS(o3d_cuda_dev, "hybrid", "long"),
        ]
    else:
        methods = [
            NNS(o3d_cuda_dev, args.search_type, "int"),
            NNS(o3d_cuda_dev, args.search_type, "long"),
        ]
    neighbors = [int(2**p) for p in range(12)]

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

                index, queries = method.setup(points, queries, radius)
                memory = measure_memory(
                    lambda: method.search(index, queries,
                                          dict(knn=knn, radius=radius)), handle)
                example_results['memory'] = memory

                results[
                    f'{example_name} n={points.shape[0]} k={knn}'] = example_results

                del index
                del points
                del queries
                o3d.core.cuda.release_cache()

        with open(f"{method}.pkl", 'wb') as f:
            pickle.dump(results, f)

    results = []
    for method in methods:
        with open(f"{method}.pkl", "rb") as f:
            print(f"{method}.pkl")
            data = pickle.load(f)
            results.append(data)

    print_system_info()
    print_table_memory(methods, results)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    dtypes = ["int32", "int64"]
    colors = ["b", "r"]
    lines = ["^", "o"]
    for idx, result in enumerate(results):  # int, long
        ks = [[], [], [], []]  # num_points
        ms = [[], [], [], []]
        for value in result.values():
            if value['num_points'] == int(np.power(10, 4)):
                ks[0].append(value['k'])
                ms[0].append(value['memory'])
            elif value['num_points'] == int(np.power(10, 5)):
                ks[1].append(value['k'])
                ms[1].append(value['memory'])
            elif value['num_points'] == int(np.power(10, 6)):
                ks[2].append(value['k'])
                ms[2].append(value['memory'])
            elif value['num_points'] == int(np.power(10, 7)):
                ks[3].append(value['k'])
                ms[3].append(value['memory'])
            else:
                raise ValueError
        ax1.plot(ks[0],
                 ms[0],
                 marker=lines[idx],
                 color=colors[idx],
                 label=dtypes[idx])
        ax2.plot(ks[1],
                 ms[1],
                 marker=lines[idx],
                 color=colors[idx],
                 label=dtypes[idx])
        ax3.plot(ks[2],
                 ms[2],
                 marker=lines[idx],
                 color=colors[idx],
                 label=dtypes[idx])
        ax4.plot(ks[3],
                 ms[3],
                 marker=lines[idx],
                 color=colors[idx],
                 label=dtypes[idx])
    ax1.set_title(f"{args.search_type}: N={int(np.power(10, 4))}")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Memory (GB)")
    ax1.legend()

    ax2.set_title(f"{args.search_type}: N={int(np.power(10, 5))}")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Memory (GB)")
    ax2.legend()

    ax3.set_title(f"{args.search_type}: N={int(np.power(10, 6))}")
    ax3.set_xlabel("K")
    ax3.set_ylabel("Memory (GB)")
    ax3.legend()

    ax4.set_title(f"{args.search_type}: N={int(np.power(10, 7))}")
    ax4.set_xlabel("K")
    ax4.set_ylabel("Memory (GB)")
    ax4.legend()

    plt.show()
    plt.savefig("memory.png")