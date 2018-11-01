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

#include "UnitTest.h"

#include "Core/Geometry/AccumulatedPoint.h"
#include "Core/Geometry/PointCloud.h"

using namespace Eigen;
using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(AccumulatedPoint, Default)
{
    int size = 100;

    Vector3d pmin(0.0, 0.0, 0.0);
    Vector3d pmax(1000.0, 1000.0, 1000.0);

    Vector3d nmin(0.0, 0.0, 0.0);
    Vector3d nmax(1.0, 1.0, 1.0);

    Vector3d cmin(0.0, 0.0, 0.0);
    Vector3d cmax(255.0, 255.0, 255.0);

    PointCloud pc;

    pc.points_.resize(size);
    pc.normals_.resize(size);
    pc.colors_.resize(size);

    Rand(pc.points_, pmin, pmax, 0);
    Rand(pc.normals_, nmin, nmax, 0);
    Rand(pc.colors_, cmin, cmax, 0);

    AccumulatedPoint accpoint;

    for (size_t i = 0; i < pc.points_.size(); i++)
        accpoint.AddPoint(pc, i);

    ExpectEQ(Vector3d(531.137254, 535.176470, 501.882352),
                        accpoint.GetAveragePoint());
    ExpectEQ(Vector3d(0.586397, 0.590857, 0.554099),
                        accpoint.GetAverageNormal());
    ExpectEQ(Vector3d(135.44, 136.47, 127.98),
                        accpoint.GetAverageColor());
}
