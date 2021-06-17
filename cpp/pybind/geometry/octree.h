// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
