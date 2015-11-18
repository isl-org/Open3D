// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include <memory>
#include "Geometry.h"

namespace three {

/// Class IGeometryOwner defines the behavior of any class that may own
/// single or multiple readonly Geometry object(s).
/// This interface is used to define auxilary data structures that rely on the
/// content of the geometry but do not change it. Examples include: Visualizer,
/// KDTree.
/// Note:
/// 1. Once AddGeometry() is called, the interface owns the geometry. As long as
/// the interface is active, the geometry must not be released. This is usually
/// implemented via keeping a copy of std::shared_ptr.
/// 2. The interface may copy data from the geometry, but may not change it.
/// 3. If an added geometry is changed, the behavior of the interface is
/// undefined. Programmers are responsible for calling UpdateGeometry() to
/// notify the interface that the geometry has been changed and the interface
/// should be updated accordingly.
class IGeometryOwner
{
public:
	virtual ~IGeometryOwner() {}
	
public:
	virtual bool AddGeometry(std::shared_ptr<const Geometry> geometry_ptr) = 0;
	virtual bool UpdateGeometry() = 0;
	virtual bool HasGeometry() const = 0;
};

}	// namespace three
