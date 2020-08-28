// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
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

#include "open3d/ml/impl/cloud/cloud.h"

namespace open3d {
namespace ml {
namespace impl {

using namespace std;

void ordered_neighbors(vector<PointXYZ>& queries,
                       vector<PointXYZ>& supports,
                       vector<int>& neighbors_indices,
                       float radius);

void batch_ordered_neighbors(vector<PointXYZ>& queries,
                             vector<PointXYZ>& supports,
                             vector<int>& q_batches,
                             vector<int>& s_batches,
                             vector<int>& neighbors_indices,
                             float radius);

void batch_nanoflann_neighbors(vector<PointXYZ>& queries,
                               vector<PointXYZ>& supports,
                               vector<int>& q_batches,
                               vector<int>& s_batches,
                               vector<int>& neighbors_indices,
                               float radius);
}  // namespace impl
}  // namespace ml
}  // namespace open3d
