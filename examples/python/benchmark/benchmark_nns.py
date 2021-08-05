import os
from subprocess import PIPE, run

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # does not affect results
import argparse
import itertools
import pickle
from collections import OrderedDict

import numpy as np
import open3d as o3d
import tabulate
from matplotlib import pyplot as plt


class O3DKnn:

    def __init__(self):
        self.nns = None

    def setup(self, points):
        self.nns = o3d.core.nns.KnnIndex()
        self.nns.set_tensor_data(points)
        return True

    def search(self, queries, knn):
        ans = self.nns.knn_search(queries, knn)
        return ans

    def clear(self):
        del self.nns


class O3DFaiss(O3DKnn):

    def setup(self, points):
        self.nns = o3d.core.nns.FaissIndex()
        self.nns.set_tensor_data(points)
        return True


def run_command(command):
    result = run(command,
                 stdout=PIPE,
                 stderr=PIPE,
                 universal_newlines=True,
                 shell=True)
    return result.stdout


def measure_time(fn, min_samples=10, max_samples=100, max_time_in_sec=100.0):
    """Measure time to run fn. Returns the elapsed time each run."""
    from time import perf_counter_ns
    t = []
    for i in range(max_samples):
        if sum(t) / 1e9 >= max_time_in_sec and i >= min_samples:
            break
        t.append(-perf_counter_ns())
        try:
            ans = fn()
        except Exception as e:
            print(e)
            return np.array([np.nan])
        t[-1] += perf_counter_ns()
        del ans
    print('.', end='')
    return np.array(t) / 1e9


def print_table(method_names, results):
    headers = [''] + [f'{n}_setup' for n in method_names
                     ] + [f'{n}_search' for n in method_names]
    rows = []

    for x in results[0]:
        r = [x] + list(
            map(np.median, [r[x]['knn_gpu_setup'] for r in results] +
                [r[x]['knn_gpu_search'] for r in results]))
        rows.append(r)

    print(tabulate.tabulate(rows, headers=headers))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--file",
                        action="append",
                        default=["/root/code/Open3D/small_tower.ply"])
    args = parser.parse_args()

    # cuda device
    o3d_cuda_dev = o3d.core.Device(o3d.core.Device.CUDA, 0)
    # collects runtimes for all examples
    results = OrderedDict()

    # setup dataset examples
    datasets = OrderedDict()

    for i, file in enumerate(args.file):
        pcd = o3d.t.io.read_point_cloud(file)
        points = queries = pcd.point['points']
        filename = os.path.basename(file)
        datasets[filename] = {'points': points, 'queries': queries}

    # random data
    points = queries = o3d.core.Tensor.from_numpy(
        np.random.rand(points.shape[0], 3).astype(np.float32))
    datasets['random'] = {'points': points, 'queries': queries}

    # prepare methods
    methods = [O3DKnn(), O3DFaiss()]
    method_names = [m.__class__.__name__ for m in methods]

    # run benchmark
    for method_name, method in zip(method_names, methods):

        print(method_name)
        if not args.overwrite and os.path.exists(f"{method_name}.pkl"):
            print(f"skip {method_name}")
            continue

        for example_name, example in datasets.items():
            print(example_name)
            points = example['points']
            queries = example['queries']

            for knn, step in itertools.product((1, 37, 64), (1, 10, 100)):
                print(knn, step)
                points = example['points']
                queries = example['queries']

                points = points[::step].contiguous().to(o3d_cuda_dev)
                queries = queries[::step].contiguous().to(o3d_cuda_dev)

                example_results = {'k': knn, 'num_points': points.shape[0]}

                ans = measure_time(lambda: method.setup(points))
                example_results['knn_gpu_setup'] = ans

                ans = measure_time(lambda: method.search(queries, knn))
                example_results['knn_gpu_search'] = ans

                method.clear()
                o3d.core.cuda.release_cache()

                results[
                    f'{example_name} n={points.shape[0]} k={knn}'] = example_results

                del points
                del queries
                o3d.core.cuda.release_cache()

    with open(f'{method_name}.pkl', 'wb') as f:
        pickle.dump(results, f)

    results = []
    for method_name, method in zip(method_names, methods):
        with open(f"{method_name}.pkl", "rb") as f:
            print(f"{method_name}.pkl")
            data = pickle.load(f)
            results.append(data)

    # print results
    nvcc_version = run_command("nvcc --version")
    os_version = run_command("cat /etc/os-release")
    cpu_info = run_command("cat /proc/cpuinfo | grep 'model name'")
    print("CUDA version")
    print(nvcc_version)
    print("")
    print("OS")
    print(os_version)
    print("")
    print("CPU")
    print(cpu_info)
    print_table(method_names, results)
