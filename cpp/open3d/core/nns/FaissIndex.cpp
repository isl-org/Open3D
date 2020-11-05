// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/core/nns/FaissIndex.h"

#include <faiss/IndexFlat.h>

#include "open3d/utility/Console.h"

#ifdef BUILD_CUDA_MODULE
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif

#include <faiss/impl/AuxIndexStructures.h>

namespace open3d {
namespace core {
namespace nns {

void TestFaissIntegration() {
    int num_dataset = 10;
    int num_query = 2;
    int num_dimension = 3;
    int knn = 3;

    std::vector<float> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0,
                              0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.2, 0.0, 0.2,
                              0.0, 0.0, 0.2, 0.1, 0.0, 0.2, 0.2, 0.1, 0.0, 0.0};

    std::vector<float> query{0.064705, 0.043921, 0.087843,
                             0.064705, 0.043921, 0.087843};

    faiss::IndexFlatL2 index(num_dimension);
    index.add(num_dataset, points.data());

    std::vector<int64_t> indices;
    std::vector<float> distances;

    indices.resize(knn * num_query);
    distances.resize(knn * num_query);

    utility::LogInfo("Search Knn on CPU.");
    index.search(num_query, query.data(), knn, distances.data(),
                 indices.data());
#ifdef BUILD_CUDA_MODULE
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatL2 gpu_index(&res, num_dimension);

    gpu_index.add(num_dataset, points.data());

    indices.clear();
    distances.clear();
    indices.resize(knn * num_query);
    distances.resize(knn * num_query);

    utility::LogInfo("Search Knn on GPU.");
    gpu_index.search(num_query, query.data(), knn, distances.data(),
                     indices.data());
#endif
}
}  // namespace nns
}  // namespace core
}  // namespace open3d
