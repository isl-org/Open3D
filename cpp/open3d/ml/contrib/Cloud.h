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
// Source code from: https://github.com/HuguesTHOMAS/KPConv.
//
// MIT License
//
// Copyright (c) 2019 HuguesTHOMAS
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//
//		0==========================0
//		|    Local feature test    |
//		0==========================0
//
//		version 1.0 :
//			>
//
//---------------------------------------------------
//
//		Cloud header
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//

#pragma once

#include <time.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace open3d {
namespace ml {
namespace contrib {

// Point class
// ***********

class PointXYZ {
public:
    // Elements
    // ********

    float x, y, z;

    // Methods
    // *******

    // Constructor
    PointXYZ() = default;

    PointXYZ(float x0, float y0, float z0) {
        x = x0;
        y = y0;
        z = z0;
    }

    // array type accessor
    float operator[](int i) const {
        if (i == 0)
            return x;
        else if (i == 1)
            return y;
        else
            return z;
    }

    // operations
    float dot(const PointXYZ P) const { return x * P.x + y * P.y + z * P.z; }

    float sq_norm() { return x * x + y * y + z * z; }

    PointXYZ cross(const PointXYZ P) const {
        return PointXYZ(y * P.z - z * P.y, z * P.x - x * P.z,
                        x * P.y - y * P.x);
    }

    PointXYZ& operator+=(const PointXYZ& P) {
        x += P.x;
        y += P.y;
        z += P.z;
        return *this;
    }

    PointXYZ& operator-=(const PointXYZ& P) {
        x -= P.x;
        y -= P.y;
        z -= P.z;
        return *this;
    }

    PointXYZ& operator*=(const float& a) {
        x *= a;
        y *= a;
        z *= a;
        return *this;
    }

    static PointXYZ floor(const PointXYZ P) {
        return PointXYZ(std::floor(P.x), std::floor(P.y), std::floor(P.z));
    }
};

// Point Operations
// *****************

inline PointXYZ operator+(const PointXYZ A, const PointXYZ B) {
    return PointXYZ(A.x + B.x, A.y + B.y, A.z + B.z);
}

inline PointXYZ operator-(const PointXYZ A, const PointXYZ B) {
    return PointXYZ(A.x - B.x, A.y - B.y, A.z - B.z);
}

inline PointXYZ operator*(const PointXYZ P, const float a) {
    return PointXYZ(P.x * a, P.y * a, P.z * a);
}

inline PointXYZ operator*(const float a, const PointXYZ P) {
    return PointXYZ(P.x * a, P.y * a, P.z * a);
}

inline std::ostream& operator<<(std::ostream& os, const PointXYZ P) {
    return os << "[" << P.x << ", " << P.y << ", " << P.z << "]";
}

inline bool operator==(const PointXYZ A, const PointXYZ B) {
    return A.x == B.x && A.y == B.y && A.z == B.z;
}

PointXYZ max_point(std::vector<PointXYZ> points);
PointXYZ min_point(std::vector<PointXYZ> points);

struct PointCloud {
    std::vector<PointXYZ> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }
};

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
