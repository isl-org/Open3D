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

#pragma once

#include <math.h>

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace ml {
namespace impl {

constexpr int NMS_BLOCK_SIZE = sizeof(uint64_t) * 8;
constexpr float EPS = 1e-8;

struct Point {
    OPEN3D_HOST_DEVICE Point() {}

    OPEN3D_HOST_DEVICE Point(double x, double y) : x(x), y(y) {}

    OPEN3D_HOST_DEVICE void set(float x, float y) {
        this->x = x;
        this->y = y;
    }

    OPEN3D_HOST_DEVICE Point operator+(const Point &b) const {
        return Point(x + b.x, y + b.y);
    }

    OPEN3D_HOST_DEVICE Point operator-(const Point &b) const {
        return Point(x - b.x, y - b.y);
    }

    float x, y;
};

OPEN3D_HOST_DEVICE inline float cross(const Point &a, const Point &b) {
    return a.x * b.y - a.y * b.x;
}

OPEN3D_HOST_DEVICE inline float cross(const Point &p1,
                                      const Point &p2,
                                      const Point &p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

OPEN3D_HOST_DEVICE inline int check_rect_cross(const Point &p1,
                                               const Point &p2,
                                               const Point &q1,
                                               const Point &q2) {
    int ret = fmin(p1.x, p2.x) <= fmax(q1.x, q2.x) &&
              fmin(q1.x, q2.x) <= fmax(p1.x, p2.x) &&
              fmin(p1.y, p2.y) <= fmax(q1.y, q2.y) &&
              fmin(q1.y, q2.y) <= fmax(p1.y, p2.y);
    return ret;
}

OPEN3D_HOST_DEVICE inline int check_in_box2d(const float *box, const Point &p) {
    // box (5): [x1, y1, x2, y2, angle]
    const float MARGIN = 1e-5;

    float center_x = (box[0] + box[2]) / 2;
    float center_y = (box[1] + box[3]) / 2;
    float angle_cos = cos(-box[4]),
          angle_sin = sin(-box[4]);  // rotate the point in the opposite
                                     // direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin +
                  center_x;
    float rot_y = -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos +
                  center_y;
    return (rot_x > box[0] - MARGIN && rot_x < box[2] + MARGIN &&
            rot_y > box[1] - MARGIN && rot_y < box[3] + MARGIN);
}

OPEN3D_HOST_DEVICE inline int intersection(const Point &p1,
                                           const Point &p0,
                                           const Point &q1,
                                           const Point &q0,
                                           Point &ans) {
    // Fast exclusion.
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

    // Check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // Calculate intersection of two lines.
    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > EPS) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x,
              c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x,
              c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

OPEN3D_HOST_DEVICE inline void rotate_around_center(const Point &center,
                                                    const float angle_cos,
                                                    const float angle_sin,
                                                    Point &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin +
                  center.x;
    float new_y = -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos +
                  center.y;
    p.set(new_x, new_y);
}

OPEN3D_HOST_DEVICE inline int point_cmp(const Point &a,
                                        const Point &b,
                                        const Point &center) {
    return atan2(a.y - center.y, a.x - center.x) >
           atan2(b.y - center.y, b.x - center.x);
}

OPEN3D_HOST_DEVICE inline float box_overlap(const float *box_a,
                                            const float *box_b) {
    // box_a (5) [x1, y1, x2, y2, angle]
    // box_b (5) [x1, y1, x2, y2, angle]

    float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 = box_a[3],
          a_angle = box_a[4];
    float b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[2], b_y2 = box_b[3],
          b_angle = box_b[4];

    Point center_a((a_x1 + a_x2) / 2, (a_y1 + a_y2) / 2);
    Point center_b((b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2);

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // Get oriented corners.
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin,
                             box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin,
                             box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // Get intersection of lines.
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                                box_b_corners[j + 1], box_b_corners[j],
                                cross_points[cnt]);
            if (flag) {
                poly_center = poly_center + cross_points[cnt];
                cnt++;
            }
        }
    }

    // Check corners.
    for (int k = 0; k < 4; k++) {
        if (check_in_box2d(box_a, box_b_corners[k])) {
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_in_box2d(box_b, box_a_corners[k])) {
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // Sort the points of polygon.
    Point temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    // Get the overlap areas.
    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        area += cross(cross_points[k] - cross_points[0],
                      cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

OPEN3D_HOST_DEVICE inline float iou_bev(const float *box_a,
                                        const float *box_b) {
    // params: box_a (5) [x1, y1, x2, y2, angle]
    // params: box_b (5) [x1, y1, x2, y2, angle]
    float sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    float sb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
