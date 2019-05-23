/* License: Apache 2.0. See LICENSE file in root directory.
   Copyright(c) 2015 Intel Corporation. All Rights Reserved. */

#ifndef LIBREALSENSE_RSUTIL_H
#define LIBREALSENSE_RSUTIL_H

#include "rs.h"
#include "assert.h"

/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
static void rs_project_point_to_pixel(float pixel[2], const struct rs_intrinsics * intrin, const float point[3])
{
    assert(intrin->model != RS_DISTORTION_INVERSE_BROWN_CONRADY); // Cannot project to an inverse-distorted image

    float x = point[0] / point[2], y = point[1] / point[2];
    if(intrin->model == RS_DISTORTION_MODIFIED_BROWN_CONRADY)
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
    pixel[0] = x * intrin->fx + intrin->ppx;
    pixel[1] = y * intrin->fy + intrin->ppy;
}

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
static void rs_deproject_pixel_to_point(float point[3], const struct rs_intrinsics * intrin, const float pixel[2], float depth)
{
    assert(intrin->model != RS_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image

    float x = (pixel[0] - intrin->ppx) / intrin->fx;
    float y = (pixel[1] - intrin->ppy) / intrin->fy;
    if(intrin->model == RS_DISTORTION_INVERSE_BROWN_CONRADY)
    {
        float r2  = x*x + y*y;
        float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
        float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
        float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
        x = ux;
        y = uy;
    }
    point[0] = depth * x;
    point[1] = depth * y;
    point[2] = depth;
}

/* Transform 3D coordinates relative to one sensor to 3D coordinates relative to another viewpoint */
static void rs_transform_point_to_point(float to_point[3], const struct rs_extrinsics * extrin, const float from_point[3])
{
    to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[3] * from_point[1] + extrin->rotation[6] * from_point[2] + extrin->translation[0];
    to_point[1] = extrin->rotation[1] * from_point[0] + extrin->rotation[4] * from_point[1] + extrin->rotation[7] * from_point[2] + extrin->translation[1];
    to_point[2] = extrin->rotation[2] * from_point[0] + extrin->rotation[5] * from_point[1] + extrin->rotation[8] * from_point[2] + extrin->translation[2];
}

/* Provide access to several recommend sets of depth control parameters */
static void rs_apply_depth_control_preset(rs_device * device, int preset)
{
    static const rs_option depth_control_options[10] = {
        RS_OPTION_R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_DECREMENT,
        RS_OPTION_R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_INCREMENT,
        RS_OPTION_R200_DEPTH_CONTROL_MEDIAN_THRESHOLD,
        RS_OPTION_R200_DEPTH_CONTROL_SCORE_MINIMUM_THRESHOLD,
        RS_OPTION_R200_DEPTH_CONTROL_SCORE_MAXIMUM_THRESHOLD,
        RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_COUNT_THRESHOLD, 
        RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_DIFFERENCE_THRESHOLD,
        RS_OPTION_R200_DEPTH_CONTROL_SECOND_PEAK_THRESHOLD,
        RS_OPTION_R200_DEPTH_CONTROL_NEIGHBOR_THRESHOLD,
        RS_OPTION_R200_DEPTH_CONTROL_LR_THRESHOLD
    };
    double depth_control_presets[6][10] = {
        {5, 5, 192,  1,  512, 6, 24, 27,  7,   24}, /* (DEFAULT)   Default settings on chip. Similiar to the medium setting and best for outdoors. */
        {5, 5,   0,  0, 1023, 0,  0,  0,  0, 2047}, /* (OFF)       Disable almost all hardware-based outlier removal */
        {5, 5, 115,  1,  512, 6, 18, 25,  3,   24}, /* (LOW)       Provide a depthmap with a lower number of outliers removed, which has minimal false negatives. */
        {5, 5, 185,  5,  505, 6, 35, 45, 45,   14}, /* (MEDIUM)    Provide a depthmap with a medium number of outliers removed, which has balanced approach. */
        {5, 5, 175, 24,  430, 6, 48, 47, 24,   12}, /* (OPTIMIZED) Provide a depthmap with a medium/high number of outliers removed. Derived from an optimization function. */
        {5, 5, 235, 27,  420, 8, 80, 70, 90,   12}, /* (HIGH)      Provide a depthmap with a higher number of outliers removed, which has minimal false positives. */
    };
    rs_set_device_options(device, depth_control_options, 10, depth_control_presets[preset], 0);
}

/* Provide access to several recommend sets of option presets for ivcam */
static void rs_apply_ivcam_preset(rs_device * device, int preset)
{
    const rs_option arr_options[15] = {
        RS_OPTION_SR300_AUTO_RANGE_ENABLE_MOTION_VERSUS_RANGE, 
        RS_OPTION_SR300_AUTO_RANGE_ENABLE_LASER,               
        RS_OPTION_SR300_AUTO_RANGE_MIN_MOTION_VERSUS_RANGE,    
        RS_OPTION_SR300_AUTO_RANGE_MAX_MOTION_VERSUS_RANGE,    
        RS_OPTION_SR300_AUTO_RANGE_START_MOTION_VERSUS_RANGE,  
        RS_OPTION_SR300_AUTO_RANGE_MIN_LASER,                  
        RS_OPTION_SR300_AUTO_RANGE_MAX_LASER,                  
        RS_OPTION_SR300_AUTO_RANGE_START_LASER,                
        RS_OPTION_SR300_AUTO_RANGE_UPPER_THRESHOLD, 
        RS_OPTION_SR300_AUTO_RANGE_LOWER_THRESHOLD,
        RS_OPTION_F200_LASER_POWER,
        RS_OPTION_F200_ACCURACY,
        RS_OPTION_F200_FILTER_OPTION,
        RS_OPTION_F200_CONFIDENCE_THRESHOLD,
        RS_OPTION_F200_MOTION_RANGE
    };

    const double arr_values[][15] = {
        {1,     1, 180,  605,  303,   2,  16,  -1, 1250, 650,  1,  1,  5,  1, -1}, /* Common                 */
        {1,     1, 180,  303,  180,   2,  16,  -1, 1000, 450,  1,  1,  5,  1, -1}, /* ShortRange             */
        {1,     0, 303,  605,  303,  -1,  -1,  -1, 1250, 975,  1,  1,  7,  0, -1}, /* LongRange              */
        {0,     0,  -1,   -1,   -1,  -1,  -1,  -1,   -1,  -1, 16,  1,  6,  0, 22}, /* BackgroundSegmentation */
        {1,     1, 100,  179,  100,   2,  16,  -1, 1000, 450,  1,  1,  6,  3, -1}, /* GestureRecognition     */
        {0,     1,  -1,   -1,   -1,   2,  16,  16, 1000, 450,  1,  1,  3,  1,  9}, /* ObjectScanning         */
        {0,     0,  -1,   -1,   -1,  -1,  -1,  -1,   -1,  -1, 16,  1,  5,  1, 22}, /* FaceMW                 */
        {2,     0,  40, 1600,  800,  -1,  -1,  -1,   -1,  -1,  1, -1, -1, -1, -1}, /* FaceLogin              */
        {1,     1, 100,  179,  179,   2,  16,  -1, 1000, 450,  1,  1,  6,  1, -1}  /* GRCursorMode           */
    };

    if(arr_values[preset][14] != -1) rs_set_device_options(device, arr_options, 15, arr_values[preset], 0);
    if(arr_values[preset][13] != -1) rs_set_device_options(device, arr_options, 14, arr_values[preset], 0);
    else rs_set_device_options(device, arr_options, 11, arr_values[preset], 0);
}

#endif
