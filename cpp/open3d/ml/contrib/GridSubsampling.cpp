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

#include "open3d/ml/contrib/GridSubsampling.h"

namespace open3d {
namespace ml {
namespace contrib {

void grid_subsampling(std::vector<PointXYZ>& original_points,
                      std::vector<PointXYZ>& subsampled_points,
                      std::vector<float>& original_features,
                      std::vector<float>& subsampled_features,
                      std::vector<int>& original_classes,
                      std::vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose) {
    // Initialize variables
    // ******************

    // Number of points in the cloud
    size_t N = original_points.size();

    // Dimension of the features
    size_t fdim = original_features.size() / N;
    size_t ldim = original_classes.size() / N;

    // Limits of the cloud
    PointXYZ minCorner = min_point(original_points);
    PointXYZ maxCorner = max_point(original_points);
    PointXYZ originCorner =
            PointXYZ::floor(minCorner * (1 / sampleDl)) * sampleDl;

    // Dimensions of the grid
    size_t sampleNX =
            (size_t)floor((maxCorner.x - originCorner.x) / sampleDl) + 1;
    size_t sampleNY =
            (size_t)floor((maxCorner.y - originCorner.y) / sampleDl) + 1;
    // size_t sampleNZ = (size_t)floor((maxCorner.z - originCorner.z) /
    // sampleDl) + 1;

    // Check if features and classes need to be processed
    bool use_feature = original_features.size() > 0;
    bool use_classes = original_classes.size() > 0;

    // Create the sampled map
    // **********************

    // Verbose parameters
    int i = 0;
    int nDisp = static_cast<int>(N / 100);

    // Initialize variables
    std::unordered_map<size_t, SampledData> data;

    for (auto& p : original_points) {
        size_t iX, iY, iZ, mapIdx;

        // Position of point in sample map
        iX = (size_t)std::floor((p.x - originCorner.x) / sampleDl);
        iY = (size_t)std::floor((p.y - originCorner.y) / sampleDl);
        iZ = (size_t)std::floor((p.z - originCorner.z) / sampleDl);
        mapIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;

        // If not already created, create key
        if (data.count(mapIdx) < 1)
            data.emplace(mapIdx, SampledData(fdim, ldim));

        // Fill the sample map
        if (use_feature && use_classes)
            data[mapIdx].update_all(p, original_features.begin() + i * fdim,
                                    original_classes.begin() + i * ldim);
        else if (use_feature)
            data[mapIdx].update_features(p,
                                         original_features.begin() + i * fdim);
        else if (use_classes)
            data[mapIdx].update_classes(p, original_classes.begin() + i * ldim);
        else
            data[mapIdx].update_points(p);

        // Display
        i++;
        if (verbose > 1 && i % nDisp == 0)
            std::cout << "\rSampled Map : " << std::setw(3) << i / nDisp << "%";
    }

    // Divide for barycentre and transfer to a vector
    subsampled_points.reserve(data.size());
    if (use_feature) subsampled_features.reserve(data.size() * fdim);
    if (use_classes) subsampled_classes.reserve(data.size() * ldim);
    for (auto& v : data) {
        subsampled_points.push_back(v.second.point * (1.0f / v.second.count));
        if (use_feature) {
            float count = (float)v.second.count;
            transform(v.second.features.begin(), v.second.features.end(),
                      v.second.features.begin(),
                      [count](float f) { return f / count; });
            subsampled_features.insert(subsampled_features.end(),
                                       v.second.features.begin(),
                                       v.second.features.end());
        }
        if (use_classes) {
            for (int i = 0; i < static_cast<int>(ldim); i++)
                subsampled_classes.push_back(
                        max_element(v.second.labels[i].begin(),
                                    v.second.labels[i].end(),
                                    [](const std::pair<int, int>& a,
                                       const std::pair<int, int>& b) {
                                        return a.second < b.second;
                                    })
                                ->first);
        }
    }

    return;
}

void batch_grid_subsampling(std::vector<PointXYZ>& original_points,
                            std::vector<PointXYZ>& subsampled_points,
                            std::vector<float>& original_features,
                            std::vector<float>& subsampled_features,
                            std::vector<int>& original_classes,
                            std::vector<int>& subsampled_classes,
                            std::vector<int>& original_batches,
                            std::vector<int>& subsampled_batches,
                            float sampleDl,
                            int max_p) {
    // Initialize variables
    // ******************

    int b = 0;
    int sum_b = 0;

    // Number of points in the cloud
    size_t N = original_points.size();

    // Dimension of the features
    size_t fdim = original_features.size() / N;
    size_t ldim = original_classes.size() / N;

    // Handle max_p = 0
    if (max_p < 1) max_p = static_cast<int>(N);

    // Loop over batches
    // *****************

    for (b = 0; b < static_cast<int>(original_batches.size()); b++) {
        // Extract batch points features and labels
        std::vector<PointXYZ> b_o_points = std::vector<PointXYZ>(
                original_points.begin() + sum_b,
                original_points.begin() + sum_b + original_batches[b]);

        std::vector<float> b_o_features;
        if (original_features.size() > 0) {
            b_o_features = std::vector<float>(
                    original_features.begin() + sum_b * fdim,
                    original_features.begin() +
                            (sum_b + original_batches[b]) * fdim);
        }

        std::vector<int> b_o_classes;
        if (original_classes.size() > 0) {
            b_o_classes =
                    std::vector<int>(original_classes.begin() + sum_b * ldim,
                                     original_classes.begin() + sum_b +
                                             original_batches[b] * ldim);
        }

        // Create result containers
        std::vector<PointXYZ> b_s_points;
        std::vector<float> b_s_features;
        std::vector<int> b_s_classes;

        // Compute subsampling on current batch
        grid_subsampling(b_o_points, b_s_points, b_o_features, b_s_features,
                         b_o_classes, b_s_classes, sampleDl, 0);

        // Stack batches points features and labels
        // ****************************************

        // If too many points remove some
        if (static_cast<int>(b_s_points.size()) <= max_p) {
            subsampled_points.insert(subsampled_points.end(),
                                     b_s_points.begin(), b_s_points.end());

            if (original_features.size() > 0)
                subsampled_features.insert(subsampled_features.end(),
                                           b_s_features.begin(),
                                           b_s_features.end());

            if (original_classes.size() > 0)
                subsampled_classes.insert(subsampled_classes.end(),
                                          b_s_classes.begin(),
                                          b_s_classes.end());

            subsampled_batches.push_back(static_cast<int>(b_s_points.size()));
        } else {
            subsampled_points.insert(subsampled_points.end(),
                                     b_s_points.begin(),
                                     b_s_points.begin() + max_p);

            if (original_features.size() > 0)
                subsampled_features.insert(subsampled_features.end(),
                                           b_s_features.begin(),
                                           b_s_features.begin() + max_p * fdim);

            if (original_classes.size() > 0)
                subsampled_classes.insert(subsampled_classes.end(),
                                          b_s_classes.begin(),
                                          b_s_classes.begin() + max_p * ldim);

            subsampled_batches.push_back(max_p);
        }

        // Stack new batch lengths
        sum_b += original_batches[b];
    }

    return;
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
