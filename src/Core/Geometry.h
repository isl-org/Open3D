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

#include <Eigen/Core>

namespace three {

class Geometry
{
public:
	enum GeometryType {
		GEOMETRY_UNKNOWN = 0,
		GEOMETRY_POINTCLOUD = 1,
		GEOMETRY_TRIANGLEMESH = 2,
		GEOMETRY_IMAGE = 3,
	};
	
public:
	virtual ~Geometry() {};
	
protected:
	Geometry(GeometryType type) : geometry_type_(type) {}

public:
	virtual bool CloneFrom(const Geometry &reference) = 0;
	virtual Eigen::Vector3d GetMinBound() const = 0;
	virtual Eigen::Vector3d GetMaxBound() const = 0;
	virtual void Clear() = 0;
	virtual bool IsEmpty() const = 0;
	virtual void Transform(const Eigen::Matrix4d & transformation) = 0;
	
	GeometryType GetGeometryType() const { return geometry_type_; }
	
private:
	GeometryType geometry_type_ = GEOMETRY_UNKNOWN;
};

}	// namespace three
