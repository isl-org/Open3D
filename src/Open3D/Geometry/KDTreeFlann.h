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

#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "Open3D/Geometry/Geometry.h"
#include "Open3D/Geometry/KDTreeSearchParam.h"
#include "Open3D/Registration/Feature.h"

namespace flann {
template <typename T>
class Matrix;
template <typename T>
struct L2;
template <typename T>
class Index;
}  // namespace flann

namespace open3d {
namespace geometry {

class KDTreeFlann {
public:
    KDTreeFlann();
    KDTreeFlann(const Eigen::MatrixXd &data);
    KDTreeFlann(const Geometry &geometry);
    KDTreeFlann(const registration::Feature &feature);
    ~KDTreeFlann();
    KDTreeFlann(const KDTreeFlann &) = delete;
    KDTreeFlann &operator=(const KDTreeFlann &) = delete;

public:
    bool SetMatrixData(const Eigen::MatrixXd &data);
    bool SetGeometry(const Geometry &geometry);
    bool SetFeature(const registration::Feature &feature);

    template <typename T>
    int Search(const T &query,
               const KDTreeSearchParam &param,
               std::vector<int> &indices,
               std::vector<double> &distance2) const;

    template <typename T>
    int SearchKNN(const T &query,
                  int knn,
                  std::vector<int> &indices,
                  std::vector<double> &distance2) const;

    template <typename T>
    int SearchRadius(const T &query,
                     double radius,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const;

    template <typename T>
    int SearchHybrid(const T &query,
                     double radius,
                     int max_nn,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const;

private:
    bool SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data);

protected:
    std::vector<double> data_;
    std::unique_ptr<flann::Matrix<double>> flann_dataset_;
    std::unique_ptr<flann::Index<flann::L2<double>>> flann_index_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};

}  // namespace geometry
}  // namespace open3d
