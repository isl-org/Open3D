// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/geometry/Octree.h"

namespace open3d {
namespace geometry {

// Trampoline classes for octree datastructures
template <class OctreeNodeBase = OctreeNode>
class PyOctreeNode : public OctreeNodeBase {
public:
    using OctreeNodeBase::OctreeNodeBase;
};

// Trampoline classes for octree datastructures
template <class OctreeLeafNodeBase = OctreeLeafNode>
class PyOctreeLeafNode : public PyOctreeNode<OctreeLeafNodeBase> {
public:
    using PyOctreeNode<OctreeLeafNodeBase>::PyOctreeNode;

    bool operator==(const OctreeLeafNode& other) const override {
        PYBIND11_OVERLOAD_PURE(bool, OctreeLeafNodeBase, other);
    };

    std::shared_ptr<OctreeLeafNode> Clone() const override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<OctreeLeafNode>,
                               OctreeLeafNodeBase, );
    };
};

}  // namespace geometry
}  // namespace open3d
