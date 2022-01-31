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

from dataclasses import dataclass
import itertools
import os
import sys

import numpy as np
import open3d as o3d
import open3d.core as o3c
from collections import OrderedDict
import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) +
    "/../../../examples/python/utility")
from open3d_benchmark import list_devices, list_float_dtypes, to_numpy_dtype

py_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
test_data_path = os.path.join(os.path.dirname(py_path), 'examples', 'test_data')
sys.path.append(os.path.join(py_path, 'benchmarks'))

from open3d_benchmark import file_downloader, unzip_data

class NNSOps:

    @staticmethod
    def knn_search(dataset, queries, nns_opt):
        index = o3c.nns.NearestNeighborSearch(dataset)
        index.knn_index()
        knn = nns_opt["knn"]
        result = index.knn_search(queries, knn)
        del index
        return result

    @staticmethod
    def radius_search(dataset, queries, nns_opt):
        radius = nns_opt["radius"]
        index = o3c.nns.NearestNeighborSearch(dataset)
        index.fixed_radius_index(radius)
        radius = nns_opt["radius"]
        result = index.fixed_radius_search(queries, radius)
        del index
        return result

    @staticmethod
    def hybrid_search(dataset, queries, nns_opt):
        radius = nns_opt["radius"]
        index = o3c.nns.NearestNeighborSearch(dataset)
        index.hybrid_index(radius)
        radius, knn = nns_opt["radius"], nns_opt["knn"]
        result = index.hybrid_search(queries, radius, knn)
        del index
        return result


def prepare_benchmark_data(dataset):

    # download data from open3d_download
    base_url = "https://github.com/isl-org/open3d_downloads/releases/download/nns/"
    out_dir = os.path.join(test_data_path, 'benchmark_data')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    file = dataset
    print("==================================")
    remote_file = os.path.join(base_url, f"{file}.zip")
    zip_file = os.path.join(out_dir, f"{file}.zip")
    ply_file = os.path.join(out_dir, file)

    if not os.path.exists(zip_file) or not os.path.exists(ply_file):
        file_downloader(remote_file, out_dir)
        unzip_data(zip_file, out_dir)
        os.remove(zip_file)

    pcd = o3d.t.io.read_point_cloud(ply_file)
    points = queries = pcd.point['positions'].to(o3d.core.Dtype.Float32)
    queries = queries[::10]
    filename = os.path.basename(ply_file)
    dataset = {'points': points, 'queries': queries}
    print("")
    return dataset


def list_sizes():
    num_points = (10000,)
    return num_points


def list_dimensions():
    dimensions = (3, 8, 16, 32)
    return dimensions


def list_datasets():
    datasets = ('canyon.ply', 'fluid_1000.ply', 'kitti_1.ply', 'kitti_2.ply', 'small_tower.ply')
    return datasets

def list_datasets():
    datasets = ('canyon.ply', 'fluid_1000.ply', 'kitti_1.ply', 'kitti_2.ply', 'small_tower.ply')
    return datasets


def list_radii_or_neighbors():
    return (1, 37, 64)


@pytest.mark.parametrize("dataset", list_datasets())
@pytest.mark.parametrize("neighbors", list_radii_or_neighbors())
@pytest.mark.parametrize("device", list_devices())
def test_knn(benchmark, dataset, neighbors, device):
    
    example_name, example = prepare_benchmark_data(dataset)
    points, queries = example['points'], example['queries']
    points = points.contiguous().to(device)
    queries = queries.contiguous().to(device)

    nns_opt = {"knn" : neighbors}
    benchmark (NNSOps.knn_search(dataset, queries, nns_opt))
    
    del points
    del queries
    o3d.core.cuda.release_cache()


@pytest.mark.parametrize("dataset", list_datasets())
@pytest.mark.parametrize("radii", list_radii_or_neighbors())
@pytest.mark.parametrize("device", list_devices())
def test_radius(benchmark, dataset, radii, device):

    example_name, example = prepare_benchmark_data(dataset)
    points, queries = example['points'], example['queries']
    points = points.contiguous().to(device)
    queries = queries.contiguous().to(device)

    nns_opt = {"radius" : radii}
    benchmark (NNSOps.knn_search(dataset, queries, nns_opt))

    del points
    del queries
    o3d.core.cuda.release_cache()


@pytest.mark.parametrize("dataset", list_datasets())
@pytest.mark.parametrize("radii", list_radii_or_neighbors())
@pytest.mark.parametrize("neighbors", list_radii_or_neighbors())
@pytest.mark.parametrize("device", list_devices())
def test_radius(benchmark, dataset, radii, neighbors, device):

    example_name, example = prepare_benchmark_data(dataset)
    points, queries = example['points'], example['queries']
    points = points.contiguous().to(device)
    queries = queries.contiguous().to(device)

    nns_opt = {"knn": neighbors, "radius" : radii}
    benchmark (NNSOps.knn_search(dataset, queries, nns_opt))

    del points
    del queries
    o3d.core.cuda.release_cache()
