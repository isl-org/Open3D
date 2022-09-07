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

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
try:
    import pcl
except ImportError:
    print("PCL is not installed.")
import torch
import torch_cluster
from scipy.spatial import cKDTree, KDTree as scipy_kdtree
from sklearn.neighbors import BallTree, KDTree as sklearn_kdtree

from benchmark_utils import measure_time, print_table_simple

OUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data")
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


class BaseModule:
    def __init__(self, device, search_type):
        self.device = device
        self.search_type = search_type
    
    def __str__(self):
        return f"{self.__class__.__name__}-{self.search_type.capitalize()}({self.device.upper()})"
# Define NNS methods
class Open3D(BaseModule):

    def __init__(self, device, search_type, index_type):
        BaseModule.__init__(self, device, search_type)
        self._device = o3d.core.Device() if device == "cpu" else o3d.core.Device(o3d.core.Device.CUDA, 0)
        self.index_type = {
            "int": o3d.core.Dtype.Int32,
            "long": o3d.core.Dtype.Int64
        }[index_type]

    def prepare_data(self, points, queries):
        points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        queries = o3d.core.Tensor(queries, dtype=o3d.core.Dtype.Float32)
        return points, queries

    def setup(self, points, queries, radius):
        points_dev = points.contiguous().to(self._device)
        queries_dev = queries.contiguous().to(self._device)
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



class PyTorchCluster(BaseModule):

    def __init__(self, device, search_type):
        BaseModule.__init__(self, device, search_type)
        self._device = torch.device(device)

    def prepare_data(self, points, queries):
        points = torch.from_numpy(points).float()
        queries = torch.from_numpy(queries).float()
        return points, queries

    def setup(self, points, queries, radius):
        points_dev = points.contiguous().to(self._device)
        queries_dev = queries.contiguous().to(self._device)
        return {'points': points_dev, 'queries': queries_dev}

    def search(self, points, queries, search_args):
        if self.search_type == "knn":
            out = torch_cluster.knn(points, queries, search_args["k"])
        elif self.search_type == "radius":
            out = torch_cluster.radius(points, queries, search_args["radius"], max_num_neighbors=100) # if # points > max, randomly pick the points.
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return out



class PCL(BaseModule):
    def __init__(self, device, search_type):
        # self.device = device
        # self.search_type = search_type
        BaseModule.__init__(self, device, search_type)

    def prepare_data(self, points, queries):
        pcd0 = pcl.PointCloud()
        pcd0.from_array(points)
        pcd1 = pcl.PointCloud()
        pcd1.from_array(queries)
        return pcd0, pcd1 

    def setup(self, points, queries, radius):
        tree = pcl.KdTreeFLANN(points)
        return {'points': tree, 'queries': queries}

    def search(self, points, queries, search_args):
        if self.search_type == "knn":
            out = points.nearest_k_search_for_cloud(queries, search_args["k"])
        elif self.search_type == "radius":
            out = points.radius_search_for_cloud(queries, search_args["radius"])
        elif self.search_type == "hybrid":
            out = points.radius_search_for_cloud(queries, search_args["radius"], search_args["k"])
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return out 


class SciPy(BaseModule):
    def __init__(self, device, search_type):
        if search_type == "hybrid":
            raise ValueError("Hybrid search is not supported in SciPy.")
        BaseModule.__init__(self, device, search_type)

    def setup(self, points, queries, radius):
        tree = scipy_kdtree(points)
        return {'points': tree, 'queries': queries}

    def search(self, points, queries, search_args):
        if self.search_type == "knn":
            out = points.query(queries, search_args["k"])
        elif self.search_type == "radius":
            out = points.query_ball_point(queries, search_args["radius"])
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return out


class SklearnKDTree(BaseModule):
    def __init__(self, device, search_type):
        if search_type == "hybrid":
            raise ValueError("Hybrid search is not supported in Sklearn.")
        BaseModule.__init__(self, device, search_type)

    def setup(self, points, queries, radius):
        tree = sklearn_kdtree(points)
        return {'points': tree, 'queries': queries}

    def search(self, points, queries, search_args):
        if self.search_type == "knn":
            out = points.query(queries, search_args["k"])
        elif self.search_type == "radius":
            out = points.query_radius(queries, search_args["radius"])
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return out


class SklearnBallTree(BaseModule):
    def __init__(self, device, search_type):
        if search_type == "hybrid":
            raise ValueError("Hybrid search is not supported in Sklearn.")
        BaseModule.__init__(self, device, search_type)

    def setup(self, points, queries, radius):
        tree = BallTree(points)
        return {'points': tree, 'queries': queries}

    def search(self, points, queries, search_args):
        if self.search_type == "knn":
            out = points.query(queries, search_args["k"])
        elif self.search_type == "radius":
            out = points.query_radius(queries, search_args["radius"])
        else:
            raise ValueError(f"{self.search_type} is not supported.")
        return out


def compute_avg_radii(points, queries, neighbors):
    """Computes the radii based on the number of neighbors"""
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    if isinstance(queries, torch.Tensor):
        queries = queries.numpy()
    tree = cKDTree(points)
    avg_radii = []
    for k in neighbors:
        dist, _ = tree.query(queries, k=k + 1)
        avg_radii.append(np.mean(dist.max(axis=-1)))
    return avg_radii


def prepare_benchmark_data(num_points, dimensions, num_queries=10, data_type=np.float32):
    # setup dataset examples
    datasets = OrderedDict()

    # random dataset
    for D in dimensions:
        for N in num_points:
            npy_file = os.path.join(OUT_DIR, f"random_D={D}_N={N}.npz")

            if not os.path.exists(npy_file):
                print(f"Generating a random dataset, random_D={D}_N={N}.npy...")
                points = np.random.randn(N, D)
                queries = np.random.randn(1000, D)
                np.savez(npy_file, points=points, queries=queries)

            print(f"Loading the random dataset, random_D={D}_N={N}.npy...")
            data = np.load(npy_file)
            points = data["points"].astype(data_type)
            queries = data["queries"].astype(data_type) 
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
    args = parser.parse_args()

    # collects runtimes for all examples
    results = OrderedDict()
    num_points = [int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]
    datasets = prepare_benchmark_data(
        num_points=num_points,
        dimensions=[3] # TODO(chrockey): higher dimension
    )
    neighbors = [8, 16, 32, 64]

    # prepare method
    methods = [
        Open3D("cpu", args.search_type, args.index_type),
        PyTorchCluster("cpu", args.search_type),
        SciPy("cpu", args.search_type),
        SklearnKDTree("cpu", args.search_type),
        SklearnBallTree("cpu", args.search_type),
        # PCL(device="cpu", search_type=args.search_type),
        Open3D("cuda", args.search_type, args.index_type),
        PyTorchCluster("cuda", args.search_type)
    ]

    # run benchmark
    for method in methods:
        if not args.overwrite and os.path.exists(os.path.join(OUT_DIR, f"{method}.pkl")):
            print(f"Skip {method}...")
            continue

        for example_name, example in datasets.items():
            points, queries = example['points'], example['queries']
            n = len(points)
            if args.search_type == "knn":
                radii = neighbors
            else:
                radii = compute_avg_radii(points, queries, neighbors)
            print(f"\n{example_name} | {n} points", end="")

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
                results[f"{example_name} n={n} k={k}"] = example_results

                for v in setup_results.values():
                    del v
                o3d.core.cuda.release_cache()
                torch.cuda.empty_cache()

        # save results
        with open(os.path.join(OUT_DIR, f"{method}.pkl"), "wb") as f:
            pickle.dump(results, f)

    # load benchmark data
    results = []
    method_names = [str(m) for m in methods]
    for method in method_names:
        with open(os.path.join(OUT_DIR, f"{method}.pkl"), "rb") as f:
            data = pickle.load(f)
            results.append(data)

    print_table_simple(methods, results)

    # save plots
    log_num_points = [np.log10(x) for x in num_points]
    for k in neighbors:
        fig = plt.figure(figsize=(10,10))
        plt.title(f"# neighbors = {k}")
        plt.xlabel("# points")
        plt.ylabel("Latency (sec)")
        
        latency = {}
        for method, result in zip(method_names, results):
            if method not in latency.keys():
                latency[method] = []
            for data in result.values():
                if data['k'] == k:
                    t = np.median(data['setup']) + np.median(data['search'])
                    latency[method].append(t)

        def assign_style(name):
            lower_name = name.lower()
            style = dict(linestyle="--" if "cpu" in lower_name else "-")
            if "open3d" in lower_name:
                style["color"] = "r"
                style["marker"]  = "o"
            elif "cluster" in lower_name:
                style["color"] = "b"
                style["marker"] = "^"
            elif "pcl" in lower_name:
                style["color"] = "g"
                style["marker"] = "+"
            return style
        for method in method_names:
            style = assign_style(method) 
            plt.plot(num_points, latency[method], label=method, **style)

        plt.semilogx()
        plt.semilogy()
        plt.legend()
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        plt.savefig(os.path.join(OUT_DIR, f"benchmark_{args.search_type}_k={k}.jpeg"), bbox_inches='tight')
