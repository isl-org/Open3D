// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Source code from: https://github.com/HuguesTHOMAS/KPConv.
//
// MIT License
//
// Copyright (c) 2019 HuguesTHOMAS
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include <cstdint>
#include <nanoflann.hpp>
#include <set>

#include "open3d/ml/contrib/Cloud.h"

namespace open3d {
namespace ml {
namespace contrib {

/// TOOD: This is a temporary function for 3DML repositiory use. In the future,
/// the native Open3D Python API should be improved and used.
///
/// Nearest neighbours within a given radius.
/// For each query point, finds a set of neighbor indices whose
/// distance is less than given radius.
/// Modifies the neighbors_indices inplace.
void ordered_neighbors(std::vector<PointXYZ>& queries,
                       std::vector<PointXYZ>& supports,
                       std::vector<int>& neighbors_indices,
                       float radius);

/// TOOD: This is a temporary function for 3DML repositiory use. In the future,
/// the native Open3D Python API should be improved and used.
///
/// Nearest neighbours withing a radius with batching.
/// queries and supports are sliced with their respective batch elements.
/// Uses nanoflann to build a KDTree and find neighbors.
void batch_nanoflann_neighbors(std::vector<PointXYZ>& queries,
                               std::vector<PointXYZ>& supports,
                               std::vector<int>& q_batches,
                               std::vector<int>& s_batches,
                               std::vector<int>& neighbors_indices,
                               float radius);
}  // namespace contrib
}  // namespace ml
}  // namespace open3d
