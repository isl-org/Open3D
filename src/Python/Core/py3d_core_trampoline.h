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

#include <Python/py3d.h>
#include <Core/Core.h>
using namespace three;

template <class GeometryBase = Geometry> class PyGeometry : public GeometryBase
{
public:
	using GeometryBase::GeometryBase;
	void Clear() override { PYBIND11_OVERLOAD_PURE(void, GeometryBase, ); }
	bool IsEmpty() const override {
		PYBIND11_OVERLOAD_PURE(bool, GeometryBase, );
	}
};

template <class Geometry3DBase = Geometry3D> class PyGeometry3D :
		public PyGeometry<Geometry3DBase>
{
public:
	using PyGeometry<Geometry3DBase>::PyGeometry;
	Eigen::Vector3d GetMinBound() const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, Geometry3DBase, );
	}
	Eigen::Vector3d GetMaxBound() const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, Geometry3DBase, );
	}
	void Transform(const Eigen::Matrix4d &transformation) override {
		PYBIND11_OVERLOAD_PURE(void, Geometry3DBase, transformation);
	}
};
