// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <math.h>

#include "open3d/Macro.h"
#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace ml {
namespace contrib {

constexpr int NMS_BLOCK_SIZE = sizeof(uint64_t) * 8;
constexpr float EPS = static_cast<float>(1e-8);

struct Point {
    OPEN3D_HOST_DEVICE Point() {}
    OPEN3D_HOST_DEVICE Point(float x, float y) : x_(x), y_(y) {}
    OPEN3D_HOST_DEVICE void set(float x, float y) {
        x_ = x;
        y_ = y;
    }
    OPEN3D_HOST_DEVICE Point operator+(const Point &b) const {
        return Point(x_ + b.x_, y_ + b.y_);
    }
    OPEN3D_HOST_DEVICE Point operator-(const Point &b) const {
        return Point(x_ - b.x_, y_ - b.y_);
    }
    float x_ = 0.0f;
    float y_ = 0.0f;
};

OPEN3D_HOST_DEVICE inline float Cross(const Point &a, const Point &b) {
    return a.x_ * b.y_ - a.y_ * b.x_;
}

OPEN3D_HOST_DEVICE inline float Cross(const Point &p1,
                                      const Point &p2,
                                      const Point &p0) {
    return (p1.x_ - p0.x_) * (p2.y_ - p0.y_) -
           (p2.x_ - p0.x_) * (p1.y_ - p0.y_);
}

OPEN3D_HOST_DEVICE inline int CheckRectCross(const Point &p1,
                                             const Point &p2,
                                             const Point &q1,
                                             const Point &q2) {
    int ret = fmin(p1.x_, p2.x_) <= fmax(q1.x_, q2.x_) &&
              fmin(q1.x_, q2.x_) <= fmax(p1.x_, p2.x_) &&
              fmin(p1.y_, p2.y_) <= fmax(q1.y_, q2.y_) &&
              fmin(q1.y_, q2.y_) <= fmax(p1.y_, p2.y_);
    return ret;
}

OPEN3D_HOST_DEVICE inline int CheckInBox2D(const float *box, const Point &p) {
    // box (5): [x1, y1, x2, y2, angle].
    const float MARGIN = static_cast<float>(1e-5);

    float center_x = (box[0] + box[2]) / 2;
    float center_y = (box[1] + box[3]) / 2;
    // Rotate the point in the opposite direction of box.
    float angle_cos = cos(-box[4]), angle_sin = sin(-box[4]);
    float rot_x = (p.x_ - center_x) * angle_cos +
                  (p.y_ - center_y) * angle_sin + center_x;
    float rot_y = -(p.x_ - center_x) * angle_sin +
                  (p.y_ - center_y) * angle_cos + center_y;
    return (rot_x > box[0] - MARGIN && rot_x < box[2] + MARGIN &&
            rot_y > box[1] - MARGIN && rot_y < box[3] + MARGIN);
}

OPEN3D_HOST_DEVICE inline int Intersection(const Point &p1,
                                           const Point &p0,
                                           const Point &q1,
                                           const Point &q0,
                                           Point &ans) {
    // Fast exclusion.
    if (CheckRectCross(p0, p1, q0, q1) == 0) return 0;

    // Check Cross standing
    float s1 = Cross(q0, p1, p0);
    float s2 = Cross(p1, q1, p0);
    float s3 = Cross(p0, q1, q0);
    float s4 = Cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // Calculate Intersection of two lines.
    float s5 = Cross(q1, p1, p0);
    if (fabs(s5 - s1) > EPS) {
        ans.x_ = (s5 * q0.x_ - s1 * q1.x_) / (s5 - s1);
        ans.y_ = (s5 * q0.y_ - s1 * q1.y_) / (s5 - s1);

    } else {
        float a0 = p0.y_ - p1.y_, b0 = p1.x_ - p0.x_,
              c0 = p0.x_ * p1.y_ - p1.x_ * p0.y_;
        float a1 = q0.y_ - q1.y_, b1 = q1.x_ - q0.x_,
              c1 = q0.x_ * q1.y_ - q1.x_ * q0.y_;
        float D = a0 * b1 - a1 * b0;

        ans.x_ = (b0 * c1 - b1 * c0) / D;
        ans.y_ = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

OPEN3D_HOST_DEVICE inline void RotateAroundCenter(const Point &center,
                                                  const float angle_cos,
                                                  const float angle_sin,
                                                  Point &p) {
    float new_x = (p.x_ - center.x_) * angle_cos +
                  (p.y_ - center.y_) * angle_sin + center.x_;
    float new_y = -(p.x_ - center.x_) * angle_sin +
                  (p.y_ - center.y_) * angle_cos + center.y_;
    p.set(new_x, new_y);
}

OPEN3D_HOST_DEVICE inline int PointCmp(const Point &a,
                                       const Point &b,
                                       const Point &center) {
    return atan2(a.y_ - center.y_, a.x_ - center.x_) >
           atan2(b.y_ - center.y_, b.x_ - center.x_);
}

OPEN3D_HOST_DEVICE inline float BoxOverlap(const float *box_a,
                                           const float *box_b) {
    // box_a (5) [x1, y1, x2, y2, angle].
    // box_b (5) [x1, y1, x2, y2, angle].
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
        RotateAroundCenter(center_a, a_angle_cos, a_angle_sin,
                           box_a_corners[k]);
        RotateAroundCenter(center_b, b_angle_cos, b_angle_sin,
                           box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // Get Intersection of lines.
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = Intersection(box_a_corners[i + 1], box_a_corners[i],
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
        if (CheckInBox2D(box_a, box_b_corners[k])) {
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (CheckInBox2D(box_b, box_a_corners[k])) {
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    OPEN3D_ASSERT(cnt != 0 && "Invalid value: cnt==0.");

    poly_center.x_ /= cnt;
    poly_center.y_ /= cnt;

    // Sort the points of polygon.
    Point temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (PointCmp(cross_points[i], cross_points[i + 1], poly_center)) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    // Get the overlap areas.
    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        area += Cross(cross_points[k] - cross_points[0],
                      cross_points[k + 1] - cross_points[0]);
    }

    return static_cast<float>(fabs(area)) / 2.0f;
}

/// (x_min, z_min, x_max, z_max, y_rotate)
OPEN3D_HOST_DEVICE inline float IoUBev2DWithMinAndMax(
        const float *box_a,
        const float *box_b,
        bool intersection_only = false) {
    // params: box_a (5) [x1, y1, x2, y2, angle].
    // params: box_b (5) [x1, y1, x2, y2, angle].
    float sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    float sb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
    float s_overlap = BoxOverlap(box_a, box_b);
    if (intersection_only) {
        return s_overlap;
    } else {
        return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
    }
}

/// (x_center, z_center, x_size, z_size, y_rotate)
OPEN3D_HOST_DEVICE inline float IoUBev2DWithCenterAndSize(
        const float *box_a,
        const float *box_b,
        bool intersection_only = false) {
    float box_a_new[5];
    box_a_new[0] = box_a[0] - box_a[2] / 2;
    box_a_new[1] = box_a[1] - box_a[3] / 2;
    box_a_new[2] = box_a[0] + box_a[2] / 2;
    box_a_new[3] = box_a[1] + box_a[3] / 2;
    box_a_new[4] = box_a[4];

    float box_b_new[5];
    box_b_new[0] = box_b[0] - box_b[2] / 2;
    box_b_new[1] = box_b[1] - box_b[3] / 2;
    box_b_new[2] = box_b[0] + box_b[2] / 2;
    box_b_new[3] = box_b[1] + box_b[3] / 2;
    box_b_new[4] = box_b[4];
    return IoUBev2DWithMinAndMax(box_a_new, box_b_new, intersection_only);
}

/// (x_center, y_max, z_center, x_size, y_size, z_size, y_rotate)
OPEN3D_HOST_DEVICE inline float IoU3DWithCenterAndSize(const float *box_a,
                                                       const float *box_b) {
    float box_a_2d[5];
    box_a_2d[0] = box_a[0];
    box_a_2d[1] = box_a[2];
    box_a_2d[2] = box_a[3];
    box_a_2d[3] = box_a[5];
    box_a_2d[4] = box_a[6];

    float box_b_2d[5];
    box_b_2d[0] = box_b[0];
    box_b_2d[1] = box_b[2];
    box_b_2d[2] = box_b[3];
    box_b_2d[3] = box_b[5];
    box_b_2d[4] = box_b[6];
    float intersection_2d = IoUBev2DWithCenterAndSize(box_a_2d, box_b_2d, true);

    float y_a_min = box_a[1] - box_a[4];
    float y_a_max = box_a[1];
    float y_b_min = box_b[1] - box_b[4];
    float y_b_max = box_b[1];
    float iw = (y_a_max < y_b_max ? y_a_max : y_b_max) -
               (y_a_min > y_b_min ? y_a_min : y_b_min);
    float iou_3d = 0;
    if (iw > 0) {
        float intersection_3d = intersection_2d * iw;
        float volume_a = box_a[3] * box_a[4] * box_a[5];
        float volume_b = box_b[3] * box_b[4] * box_b[5];
        float union_3d = volume_a + volume_b - intersection_3d;
        iou_3d = intersection_3d / union_3d;
    }
    return iou_3d;
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
