/* License: Apache 2.0. See LICENSE file in root directory.
   Copyright(c) 2015 Intel Corporation. All Rights Reserved. */

#ifndef LIBREALSENSE_RSUTIL2_H
#define LIBREALSENSE_RSUTIL2_H

#include "h/rs_types.h"
#include "h/rs_sensor.h"
#include "h/rs_frame.h"
#include "rs.h"
#include "assert.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
static void rs2_project_point_to_pixel(float pixel[2], const struct rs2_intrinsics * intrin, const float point[3])
{
    float x = point[0] / point[2], y = point[1] / point[2];

    if ((intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) ||
        (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY))
    {

        float r2  = x*x + y*y;
        float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
        x *= f;
        y *= f;
        float dx = x + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
        float dy = y + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
        x = dx;
        y = dy;
    }
    if (intrin->model == RS2_DISTORTION_FTHETA)
    {
        float r = sqrtf(x*x + y*y);
        if (r < FLT_EPSILON)
        {
            r = FLT_EPSILON;
        }
        float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r* tan(intrin->coeffs[0] / 2.0f)));
        x *= rd / r;
        y *= rd / r;
    }
    if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
    {
        float r = sqrtf(x*x + y*y);
        if (r < FLT_EPSILON)
        {
            r = FLT_EPSILON;
        }
        float theta = atan(r);
        float theta2 = theta*theta;
        float series = 1 + theta2*(intrin->coeffs[0] + theta2*(intrin->coeffs[1] + theta2*(intrin->coeffs[2] + theta2*intrin->coeffs[3])));
        float rd = theta*series;
        x *= rd / r;
        y *= rd / r;
    }

    pixel[0] = x * intrin->fx + intrin->ppx;
    pixel[1] = y * intrin->fy + intrin->ppy;
}

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics * intrin, const float pixel[2], float depth)
{
    assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
    //assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

    float x = (pixel[0] - intrin->ppx) / intrin->fx;
    float y = (pixel[1] - intrin->ppy) / intrin->fy;
    if(intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
    {
        float r2  = x*x + y*y;
        float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
        float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
        float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
        x = ux;
        y = uy;
    }
    if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
    {
        float rd = sqrtf(x*x + y*y);
        if (rd < FLT_EPSILON)
        {
            rd = FLT_EPSILON;
        }

        float theta = rd;
        float theta2 = rd*rd;
        for (int i = 0; i < 4; i++)
        {
            float f = theta*(1 + theta2*(intrin->coeffs[0] + theta2*(intrin->coeffs[1] + theta2*(intrin->coeffs[2] + theta2*intrin->coeffs[3])))) - rd;
            if (abs(f) < FLT_EPSILON)
            {
                break;
            }
            float df = 1 + theta2*(3 * intrin->coeffs[0] + theta2*(5 * intrin->coeffs[1] + theta2*(7 * intrin->coeffs[2] + 9 * theta2*intrin->coeffs[3])));
            theta -= f / df;
            theta2 = theta*theta;
        }
        float r = tan(theta);
        x *= r / rd;
        y *= r / rd;
    }
    if (intrin->model == RS2_DISTORTION_FTHETA)
    {
        float rd = sqrtf(x*x + y*y);
        if (rd < FLT_EPSILON)
        {
            rd = FLT_EPSILON;
        }
        float r = (float)(tan(intrin->coeffs[0] * rd) / atan(2 * tan(intrin->coeffs[0] / 2.0f)));
        x *= r / rd;
        y *= r / rd;
    }

    point[0] = depth * x;
    point[1] = depth * y;
    point[2] = depth;
}

/* Transform 3D coordinates relative to one sensor to 3D coordinates relative to another viewpoint */
static void rs2_transform_point_to_point(float to_point[3], const struct rs2_extrinsics * extrin, const float from_point[3])
{
    to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[3] * from_point[1] + extrin->rotation[6] * from_point[2] + extrin->translation[0];
    to_point[1] = extrin->rotation[1] * from_point[0] + extrin->rotation[4] * from_point[1] + extrin->rotation[7] * from_point[2] + extrin->translation[1];
    to_point[2] = extrin->rotation[2] * from_point[0] + extrin->rotation[5] * from_point[1] + extrin->rotation[8] * from_point[2] + extrin->translation[2];
}

/* Calculate horizontal and vertical feild of view, based on video intrinsics */
static void rs2_fov(const struct rs2_intrinsics * intrin, float to_fov[2])
{
    to_fov[0] = (atan2f(intrin->ppx + 0.5f, intrin->fx) + atan2f(intrin->width - (intrin->ppx + 0.5f), intrin->fx)) * 57.2957795f;
    to_fov[1] = (atan2f(intrin->ppy + 0.5f, intrin->fy) + atan2f(intrin->height - (intrin->ppy + 0.5f), intrin->fy)) * 57.2957795f;
}

static void next_pixel_in_line(float curr[2], const float start[2], const float end[2])
{
    float line_slope = (end[1] - start[1]) / (end[0] - start[0]);
    if (fabs(end[0] - curr[0]) > fabs(end[1] - curr[1]))
    {
        curr[0] = end[0] > curr[0] ? curr[0] + 1 : curr[0] - 1;
        curr[1] = end[1] - line_slope * (end[0] - curr[0]);
    }
    else
    {
        curr[1] = end[1] > curr[1] ? curr[1] + 1 : curr[1] - 1;
        curr[0] = end[0] - ((end[1] + curr[1]) / line_slope);
    }
}

static bool is_pixel_in_line(const float curr[2], const float start[2], const float end[2])
{
    return ((end[0] >= start[0] && end[0] >= curr[0] && curr[0] >= start[0]) || (end[0] <= start[0] && end[0] <= curr[0] && curr[0] <= start[0])) &&
           ((end[1] >= start[1] && end[1] >= curr[1] && curr[1] >= start[1]) || (end[1] <= start[1] && end[1] <= curr[1] && curr[1] <= start[1]));
}

static void adjust_2D_point_to_boundary(float p[2], int width, int height)
{
    if (p[0] < 0) p[0] = 0;
    if (p[0] > width) p[0] = (float)width;
    if (p[1] < 0) p[1] = 0;
    if (p[1] > height) p[1] = (float)height;
}

/* Find projected pixel with unknown depth search along line. */
static void rs2_project_color_pixel_to_depth_pixel(float to_pixel[2],
    const uint16_t* data, float depth_scale,
    float depth_min, float depth_max,
    const struct rs2_intrinsics* depth_intrin,
    const struct rs2_intrinsics* color_intrin,
    const struct rs2_extrinsics* color_to_depth,
    const struct rs2_extrinsics* depth_to_color,
    const float from_pixel[2])
{
    //Find line start pixel
    float start_pixel[2] = { 0 }, min_point[3] = { 0 }, min_transformed_point[3] = { 0 };
    rs2_deproject_pixel_to_point(min_point, color_intrin, from_pixel, depth_min);
    rs2_transform_point_to_point(min_transformed_point, color_to_depth, min_point);
    rs2_project_point_to_pixel(start_pixel, depth_intrin, min_transformed_point);
    adjust_2D_point_to_boundary(start_pixel, depth_intrin->width, depth_intrin->height);

    //Find line end depth pixel
    float end_pixel[2] = { 0 }, max_point[3] = { 0 }, max_transformed_point[3] = { 0 };
    rs2_deproject_pixel_to_point(max_point, color_intrin, from_pixel, depth_max);
    rs2_transform_point_to_point(max_transformed_point, color_to_depth, max_point);
    rs2_project_point_to_pixel(end_pixel, depth_intrin, max_transformed_point);
    adjust_2D_point_to_boundary(end_pixel, depth_intrin->width, depth_intrin->height);

    //search along line for the depth pixel that it's projected pixel is the closest to the input pixel
    float min_dist = -1;
    for (float p[2] = { start_pixel[0], start_pixel[1] }; is_pixel_in_line(p, start_pixel, end_pixel); next_pixel_in_line(p, start_pixel, end_pixel))
    {
        float depth = depth_scale * data[(int)p[1] * depth_intrin->width + (int)p[0]];
        if (depth == 0)
            continue;

        float projected_pixel[2] = { 0 }, point[3] = { 0 }, transformed_point[3] = { 0 };
        rs2_deproject_pixel_to_point(point, depth_intrin, p, depth);
        rs2_transform_point_to_point(transformed_point, depth_to_color, point);
        rs2_project_point_to_pixel(projected_pixel, color_intrin, transformed_point);

        float new_dist = pow((projected_pixel[1] - from_pixel[1]), 2) + pow((projected_pixel[0] - from_pixel[0]), 2);
        if (new_dist < min_dist || min_dist < 0)
        {
            min_dist = new_dist;
            to_pixel[0] = p[0];
            to_pixel[1] = p[1];
        }
    }
}

#endif
