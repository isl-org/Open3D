// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include <Eigen/Dense>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <unordered_set>

#include "open3d/geometry/PlanarPatch.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace geometry {

namespace {

/// \class BoundaryVolumeHierarchy
///
/// \brief Breadth-first octree data structure
///
/// BoundaryVolumeHierarchy is different than Octree because it partitions
/// space on-demand in a breadth-first fashion rather than all at once in
/// depth-first order. Instead of specifying max_depth, this BVH can stop
/// partitioning once a node has less than min_points associated with it.
/// These features make BoundaryVolumeHierarchy more amenable to efficiently
/// detecting planes in hierarchical subregions of the point cloud.
class BoundaryVolumeHierarchy {
public:
    static constexpr int DIMENSION = 3;
    static constexpr size_t NUM_CHILDREN = 8;

    /// \brief Constructor for the root node of the octree.
    ///
    /// \param pointCloud is the associated set of points being partitioned
    BoundaryVolumeHierarchy(const PointCloud* pointCloud,
                            size_t min_points = 1,
                            double min_size = 0.0)
        : pointcloud_(pointCloud),
          min_points_(min_points),
          min_size_(min_size),
          leaf_(true),
          level_(0) {
        // set origin of root node and size of each child node (cubes)
        const Eigen::Vector3d min_bound = pointCloud->GetMinBound();
        const Eigen::Vector3d max_bound = pointCloud->GetMaxBound();
        center_ = (min_bound + max_bound) / 2;
        const Eigen::Vector3d half_sizes = center_ - min_bound;
        size_ = 2 * half_sizes.maxCoeff();

        // since this is the root, all the point cloud's indices are contained
        indices_ = std::vector<size_t>(pointCloud->points_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
    }

    /// \brief Partition a leaf node's points into NUM_CHILDREN subdivisions
    void partition() {
        // Nothing to do if already partitioned
        if (!leaf_) return;

        // size of each child
        const double child_size = size_ / 2;

        // Does this node have enough data to be able to partition further
        if (indices_.size() <= min_points_ || child_size < min_size_ ||
            indices_.size() < 2)
            return;

        // split points and create children
        for (const size_t& pidx : indices_) {
            // calculate child index comparing position to child center
            const size_t cidx = calculateChildIndex(pointcloud_->points_[pidx]);
            // if child does not yet exist, create and initialize
            if (children_[cidx] == nullptr) {
                const Eigen::Vector3d child_center =
                        calculateChildCenter(cidx, child_size);
                children_[cidx].reset(new BoundaryVolumeHierarchy(
                        pointcloud_, level_ + 1, child_center, child_size));
                children_[cidx]->indices_.reserve(indices_.size());
            }
            children_[cidx]->indices_.push_back(pidx);
        }

        // now that I have children, I am no longer a leaf node
        leaf_ = false;
        // for space efficiency, get rid of my list of points, which was
        // redistributed to my children
        indices_.clear();
    }

public:
    const PointCloud* pointcloud_;
    std::array<std::shared_ptr<BoundaryVolumeHierarchy>, NUM_CHILDREN>
            children_;
    Eigen::Vector3d center_;
    size_t min_points_;
    double min_size_;
    double size_;
    bool leaf_;
    size_t level_;
    std::vector<size_t> indices_;

private:
    /// \brief Private constructor for creating children.
    ///
    /// \param pointCloud is the (original) set of points being partitioned
    /// \param level in tree that this child lives on
    /// \param center coordinate of this child node
    /// \param size of this child (same for all nodes at this level)
    BoundaryVolumeHierarchy(const PointCloud* pointcloud,
                            size_t level,
                            const Eigen::Vector3d& center,
                            double size)
        : pointcloud_(pointcloud),
          center_(center),
          size_(size),
          leaf_(true),
          level_(level) {}

    /// \brief Calculate the center coordinate of a child node.
    ///
    /// For a root node with center_ == (0, 0, 0) and size_ == 2,
    /// child_size == 1 and:
    ///   child_index 0: center == (-0.5, -0.5, -0.5)
    ///   child_index 1: center == (-0.5, -0.5,  0.5)
    ///   child_index 2: center == (-0.5,  0.5, -0.5)
    ///   child_index 3: center == (-0.5,  0.5,  0.5)
    ///   child_index 4: center == ( 0.5, -0.5, -0.5)
    ///   child_index 5: center == ( 0.5, -0.5,  0.5)
    ///   child_index 6: center == ( 0.5,  0.5, -0.5)
    ///   child_index 7: center == ( 0.5,  0.5,  0.5)
    ///
    /// \param child_index indicates which child
    /// \param child_size of the child's bounding volume (cube)
    Eigen::Vector3d calculateChildCenter(size_t child_index,
                                         double child_size) const {
        Eigen::Vector3d center;
        for (size_t d = 0; d < DIMENSION; d++) {
            const int signal = (((child_index & (1 << (DIMENSION - d - 1))) >>
                                 (DIMENSION - d - 1))
                                << 1) -
                               1;
            center(d) = center_(d) + (child_size / 2.) * signal;
        }
        return center;
    }

    /// \brief Calculate child index given a position
    ///
    /// \param position of point to find child index of
    size_t calculateChildIndex(const Eigen::Vector3d& position) const {
        size_t child_index = 0;
        for (size_t d = 0; d < DIMENSION; d++) {
            child_index |= (position(d) > center_(d)) << (DIMENSION - d - 1);
        }
        return child_index;
    }
};

using BoundaryVolumeHierarchyPtr = std::shared_ptr<BoundaryVolumeHierarchy>;
using PlanarPatchPtr = std::shared_ptr<PlanarPatch>;

bool SplitAndDetectPatchesRecursive(const BoundaryVolumeHierarchyPtr& node,
                                    size_t min_num_points,
                                    std::vector<PlanarPatchPtr>& patches) {
    // if there aren't enough points to find a good plane, don't even try
    if (node->indices_.size() < min_num_points) return false;

    bool node_has_patch = false;
    bool child_has_patch = false;

    for (const auto& child : node->children_) {
        if (child != nullptr &&
            SplitAndDetectPatchesRecursive(child, min_num_points, patches)) {
            child_has_patch = true;
        }
    }

    if (!child_has_patch /*&& node->level_ > 2*/) {
        // if (is_planar()) {
        // node_has_patch = true;
        // patches.emplace_back();
        // }
    }

    return node_has_patch || child_has_patch;
}

}  // unnamed namespace

std::vector<std::shared_ptr<PlanarPatch>> PointCloud::DetectPlanarPatches()
        const {
    if (!HasNormals()) {
        utility::LogError(
                "DetectPlanarPatches requires pre-computed normal vectors.");
        return {};
    }

    int min_num_points = 1;

    BoundaryVolumeHierarchyPtr root =
            std::make_shared<BoundaryVolumeHierarchy>(this);
    std::vector<PlanarPatchPtr> patches;
    SplitAndDetectPatchesRecursive(root, min_num_points, patches);

    // DetectPlanarPatchesRecursive

    // split phase

    // robust planarity test

    // grow phase

    // merge phase

    // iterative grow-merge

    return {};
}

}  // namespace geometry
}  // namespace open3d
