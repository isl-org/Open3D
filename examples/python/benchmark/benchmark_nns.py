import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # does not affect results

import argparse
import itertools
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import open3d as o3d

pwd = Path(os.path.dirname(os.path.realpath(__file__)))
open3d_root = pwd.parent.parent.parent

from benchmark_utils import measure_time, print_system_info, print_table


# Define NNS methods
class O3DKnn:

    def __init__(self):
        pass

    def prepare_data(self, *args):
        return args

    def setup(self, points):
        nns = o3d.core.nns.KnnIndex()
        nns.set_tensor_data(points)
        return nns

    def search(self, nns, queries, knn):
        ans = nns.knn_search(queries, knn)
        return ans


class O3DKnnCPU(O3DKnn):

    def prepare_data(self, points, queries):
        points_cpu = points.cpu()
        queries_cpu = queries.cpu()
        return points_cpu, queries_cpu

    def setup(self, points):
        nns = o3d.core.nns.NearestNeighborSearch(points)
        nns.knn_index()
        return nns


class O3DFaiss(O3DKnn):

    def setup(self, points):
        nns = o3d.core.nns.FaissIndex()
        nns.set_tensor_data(points)
        return nns


class NativeFaiss:

    @staticmethod
    def swig_ptr_from_FloatTensor(x):
        assert x.is_contiguous()
        assert x.dtype == torch.float32
        return faiss.cast_integer_to_float_ptr(x.storage().data_ptr() +
                                               x.storage_offset() * 4)

    @staticmethod
    def swig_ptr_from_LongTensor(x):
        assert x.is_contiguous()
        assert x.dtype == torch.int64, "dtype=%s" % x.dtype
        return faiss.cast_integer_to_idx_t_ptr(x.storage().data_ptr() +
                                               x.storage_offset() * 8)

    @staticmethod
    def search_index_pytorch(index, x, k, D=None, I=None):
        """call the search function of an index with pytorch tensor I/O (CPU
            and GPU supported)"""
        assert x.is_contiguous()
        n, d = x.size()
        assert d == index.d

        if D is None:
            D = torch.empty((n, k), dtype=torch.float32, device=x.device)
        else:
            assert D.size() == (n, k)

        if I is None:
            I = torch.empty((n, k), dtype=torch.int64, device=x.device)
        else:
            assert I.size() == (n, k)
        torch.cuda.synchronize()
        xptr = NativeFaiss.swig_ptr_from_FloatTensor(x)
        Iptr = NativeFaiss.swig_ptr_from_LongTensor(I)
        Dptr = NativeFaiss.swig_ptr_from_FloatTensor(D)
        index.search_c(n, xptr, k, Dptr, Iptr)
        torch.cuda.synchronize()
        return D, I

    def __init__(self, d=3):
        self.res = faiss.StandardGpuResources()
        self.d = d

    def prepare_data(self, points, queries):
        points_np = points.cpu().numpy()
        queries_np = queries.cpu().numpy()
        return points_np, queries_np

    def setup(self, points):
        print("setup")
        index_cpu = faiss.IndexFlatL2(self.d)
        index = faiss.index_cpu_to_gpu(self.res, 0, index_cpu)
        index.add(points)
        return index

    def search(self, index, queries, knn):
        print("search")
        result = index.search(queries, knn)
        return result


class NativeFaissIVF(NativeFaiss):

    def __init__(self, d=3, nlist=100, nprobe=5):
        super().__init__(d)
        self.nlist = nlist
        self.nprobe = nprobe

    def setup(self, points):
        print("setup")
        quantizer = faiss.IndexFlatL2(self.d)  # the other index
        index_cpu = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
        # index_cpu.nprobe = self.nprobe
        index = faiss.index_cpu_to_gpu(self.res, 0, index_cpu)
        index.train(points)
        index.add(points)
        return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--file",
                        action="append",
                        default=[str(open3d_root / "small_tower.ply")])
    parser.add_argument("--test_faiss", action="store_true")
    args = parser.parse_args()

    # cuda device
    o3d_cuda_dev = o3d.core.Device(o3d.core.Device.CUDA, 0)
    # collects runtimes for all examples
    results = OrderedDict()

    # setup dataset examples
    datasets = OrderedDict()

    # TODO: remove hard-coded file list.
    files = [
        "small_tower.ply", "kitti_1.ply", "kitti_2.ply", "fluid_1000.ply",
        "s3dis_1.ply", "s3dis_2.ply"
    ]
    for i, file in enumerate(files):
        filepath = os.path.join(open3d_root, file)
        pcd = o3d.t.io.read_point_cloud(filepath)
        points = queries = pcd.point['points']
        filename = os.path.basename(filepath)
        datasets[filename] = {'points': points, 'queries': queries}

    # random data
    points = queries = o3d.core.Tensor.from_numpy(
        np.random.rand(points.shape[0], 3).astype(np.float32))
    datasets['random'] = {'points': points, 'queries': queries}

    # prepare methods
    methods = [O3DKnn(), O3DFaiss(), O3DKnnCPU()]
    if args.test_faiss:
        try:
            # Requires faiss and torch.
            # conda install faiss=1.6.5 pytorch -c pytorch
            # NOTE: faiss >= 1.7 has some error related with index other than IndexFlat
            # see https://github.com/facebookresearch/faiss/issues/1771
            import faiss
            import torch
            import torch.utils.dlpack
            methods.append(NativeFaiss())
            methods.append(NativeFaissIVF())
        except ImportError as e:
            print("faiss is not available. Please install faiss first.")

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

                points, queries = method.prepare_data(points, queries)

                ans = measure_time(lambda: method.setup(points))
                example_results['knn_gpu_setup'] = ans

                index = method.setup(points)

                ans = measure_time(lambda: method.search(index, queries, knn))
                example_results['knn_gpu_search'] = ans

                del index
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

    print_system_info()
    print_table(method_names, results)
