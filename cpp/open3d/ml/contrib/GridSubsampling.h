// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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

#include <cstdint>
#include <set>

#include "open3d/ml/contrib/Cloud.h"

namespace open3d {
namespace ml {
namespace contrib {

class SampledData {
public:
    // Elements
    // ********

    int count;
    PointXYZ point;
    std::vector<float> features;
    std::vector<std::unordered_map<int, int>> labels;

    // Methods
    // *******

    // Constructor
    SampledData() {
        count = 0;
        point = PointXYZ();
    }

    SampledData(const size_t fdim, const size_t ldim) {
        count = 0;
        point = PointXYZ();
        features = std::vector<float>(fdim);
        labels = std::vector<std::unordered_map<int, int>>(ldim);
    }

    // Method Update
    void update_all(const PointXYZ p,
                    std::vector<float>::iterator f_begin,
                    std::vector<int>::iterator l_begin) {
        count += 1;
        point += p;
        transform(features.begin(), features.end(), f_begin, features.begin(),
                  std::plus<float>());
        int i = 0;
        for (std::vector<int>::iterator it = l_begin;
             it != l_begin + labels.size(); ++it) {
            labels[i][*it] += 1;
            i++;
        }
        return;
    }

    void update_features(const PointXYZ p,
                         std::vector<float>::iterator f_begin) {
        count += 1;
        point += p;
        transform(features.begin(), features.end(), f_begin, features.begin(),
                  std::plus<float>());
        return;
    }

    void update_classes(const PointXYZ p, std::vector<int>::iterator l_begin) {
        count += 1;
        point += p;
        int i = 0;
        for (std::vector<int>::iterator it = l_begin;
             it != l_begin + labels.size(); ++it) {
            labels[i][*it] += 1;
            i++;
        }
        return;
    }

    void update_points(const PointXYZ p) {
        count += 1;
        point += p;
        return;
    }
};

void grid_subsampling(std::vector<PointXYZ>& original_points,
                      std::vector<PointXYZ>& subsampled_points,
                      std::vector<float>& original_features,
                      std::vector<float>& subsampled_features,
                      std::vector<int>& original_classes,
                      std::vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose);

void batch_grid_subsampling(std::vector<PointXYZ>& original_points,
                            std::vector<PointXYZ>& subsampled_points,
                            std::vector<float>& original_features,
                            std::vector<float>& subsampled_features,
                            std::vector<int>& original_classes,
                            std::vector<int>& subsampled_classes,
                            std::vector<int>& original_batches,
                            std::vector<int>& subsampled_batches,
                            float sampleDl,
                            int max_p);

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
