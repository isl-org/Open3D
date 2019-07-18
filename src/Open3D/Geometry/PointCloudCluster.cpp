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

#include "Open3D/Geometry/PointCloud.h"

#include <Eigen/Dense>
#include <unordered_set>

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

std::vector<int> PointCloud::ClusterDBSCAN(double eps,
                                           size_t min_points) const {
    KDTreeFlann kdtree(*this);
    // set all labels to undefined (-2)
    std::vector<int> labels(points_.size(), -2);
    int cluster_label = 0;
    for (size_t idx = 0; idx < points_.size(); ++idx) {
        if (labels[idx] != -2) {  // label is not undefined
            continue;
        }

        // find neighbours
        std::vector<int> nbs;
        std::vector<double> dists2;
        kdtree.SearchRadius(points_[idx], eps, nbs, dists2);

        // check density
        if (nbs.size() < min_points) {
            labels[idx] = -1;
            continue;
        }

        std::unordered_set<int> nbs_next(nbs.begin(), nbs.end());
        std::unordered_set<int> nbs_visited;
        nbs_visited.insert(int(idx));

        labels[idx] = cluster_label;
        while (!nbs_next.empty()) {
            int nb = *nbs_next.begin();
            nbs_next.erase(nbs_next.begin());
            nbs_visited.insert(nb);

            if (labels[nb] == -1) {  // noise label
                labels[nb] = cluster_label;
            }
            if (labels[nb] != -2) {  // not undefined label
                continue;
            }
            labels[nb] = cluster_label;

            kdtree.SearchRadius(points_[nb], eps, nbs, dists2);
            if (nbs.size() >= min_points) {
                for (int qnb : nbs) {
                    if (nbs_visited.count(qnb) == 0) {
                        nbs_next.insert(qnb);
                    }
                }
            }
        }

        cluster_label++;
    }
    return labels;
}

}  // namespace geometry
}  // namespace open3d
