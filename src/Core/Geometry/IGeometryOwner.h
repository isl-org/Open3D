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
#include <Core/Geometry/Geometry.h>

namespace three {

/// Class IGeometryOwner defines the behavior of a class that owns single or 
/// multiple readonly Geometry object(s).
/// Using this interface is discouraged unless there is a very good reason the
/// data structure must own the geometry. This interface requires the input to
/// be a smart pointer of a geometry object (instead of a local variable or a 
/// reference). The smart pointer semantics can add complexity to the code. 
/// Always think of alternative solutions like value copying before using this
/// interface.
///
/// Example:
/// Visualizer is a good example of this interface. It needs to be able to
/// efficiently respond to the changes of the geometries. Thus value copying
/// semantics is infeasible. Additionally, Visualizer needs to own the geometry
/// since it needs to access the geometry data when a new shader is being bind.
/// There is no guarantee that the lifetime of a Visualizer is always less than
/// that of the geometries it owns. Thus, it has to be an IGeometryOwner.
///
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
