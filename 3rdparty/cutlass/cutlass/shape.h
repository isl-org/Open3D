/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Defines Shape implementing the Layout concept for representing a 4D hypercube of objects.
*/
#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/*!@defgroup layout_concept Layout Concept
* @{
* @par Implementations of \ref layout_concept are used to describe a cube with DxHxW elements and C
scalars per element.
 A HxW slice of a cube is called an image and a cube consists of D images.
*
* @par Notations
*   Let Layout be an implementation of the \ref layout_concept.
*
* @par Valid Expressions
* - <b>Layout::D</b> specifies the depth of a cube
* - <b>Layout::H</b> specifies the height of a cube
* - <b>Layout::W</b> specifies the height of a cube
* - <b>Layout::C</b> specifies the number of channels of each element in a cube
* - <b>Layout::W_c</b> specifies the number of scalars of each row in one image of a cube.
* - <b>Layout::H_w</b> specifies the number of elements in an image slice.
* - <b>Layout::H_w_c</b>_specifies the number of scalars in an image slice.
* - <b>Layout::D_h_w</b> specifies the number of elements in a cube.
* - <b>Layout::D_h_w_c</b> specifies the number of scalars in a cube.
* - <b>Layout::Strides</b> is a \ref layout_concept specifying the strides.
* @}
*/

/**
* @brief A Shape implementing \ref layout_concept describing the dimensions of a cube.
* @concept{layout_concept}
*/
template <int kD_ = 1, int kH_ = 1, int kW_ = 1, int kC_ = 1>
struct Shape {
  /// The depth of the cube.
  static int const kD = kD_;
  /// The height of the cube.
  static int const kH = kH_;
  /// The width of the cube.
  static int const kW = kW_;
  /// The number of scalars per element.
  static int const kC = kC_;
};


/**
* @brief A Shape implementing \ref layout_concept describing the dimensions of a cube.
* @concept{layout_concept}
*/
template <int kH_, int kW_>
struct Shape<1, kH_, kW_, 1> {
  /// The depth of the cube.
  static int const kD = 1;
  /// The height of the cube.
  static int const kH = kH_;
  /// The width of the cube.
  static int const kW = kW_;
  /// The number of scalars per element.
  static int const kC = 1;
};

/**
* @brief Compute derived counted of a \ref layout_concept based class
*/
template <typename Shape>
struct ShapeCount {
  /// The number of elements per row.
  static int const kWc = Shape::kW * Shape::kC;
  /// The number of pixels per image.
  static int const kHw = Shape::kH * Shape::kW;
  /// The number of elements per image.
  static int const kHwc = Shape::kH * kWc;
  /// The number of pixels per cube.
  static int const kDhw = Shape::kD * kHw;
  /// The number of elements in the 4D space.
  static int const kDhwc = Shape::kD * kHwc;
  /// The number of elements in the 4D space.
  static int const kCount = kDhwc;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A_, int kScale_>
struct ShapeScale {
  typedef Shape<A_::kD * kScale_, A_::kH * kScale_, A_::kW * kScale_, A_::kC * kScale_> Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A_, typename B_>
struct ShapeAdd {
  typedef Shape<A_::kD + B_::kD, A_::kH + B_::kH, A_::kW + B_::kW, A_::kC + B_::kC> Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A_, typename B_>
struct ShapeSub {
  typedef Shape<A_::kD - B_::kD, A_::kH - B_::kH, A_::kW - B_::kW, A_::kC - B_::kC> Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A_, typename B_>
struct ShapeMul {
  typedef Shape<A_::kD * B_::kD, A_::kH * B_::kH, A_::kW * B_::kW, A_::kC * B_::kC> Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A_, typename B_>
struct ShapeDiv {
  typedef Shape<A_::kD / B_::kD, A_::kH / B_::kH, A_::kW / B_::kW, A_::kC / B_::kC> Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A_, typename B_>
struct ShapeDivCeiling {
  typedef Shape<(A_::kD + B_::kD - 1) / B_::kD,
                (A_::kH + B_::kH - 1) / B_::kH,
                (A_::kW + B_::kW - 1) / B_::kW,
                (A_::kC + B_::kC - 1) / B_::kC>
      Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A_, typename B_>
struct ShapeMax {
  typedef Shape<(A_::kD > B_::kD ? A_::kD : B_::kD),
                (A_::kH > B_::kH ? A_::kH : B_::kH),
                (A_::kW > B_::kW ? A_::kW : B_::kW),
                (A_::kC > B_::kC ? A_::kC : B_::kC)>
      Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A_, typename B_>
struct ShapeMin {
  typedef Shape<(A_::kD < B_::kD ? A_::kD : B_::kD),
                (A_::kH < B_::kH ? A_::kH : B_::kH),
                (A_::kW < B_::kW ? A_::kW : B_::kW),
                (A_::kC < B_::kC ? A_::kC : B_::kC)>
      Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Shape_, int elementsPerAccess>
struct ShapeStrides {
  typedef Shape<Shape_::kH * Shape_::kW * Shape_::kC,
                Shape_::kW * Shape_::kC,
                Shape_::kC,
                elementsPerAccess>
      Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief Compute the offset for the given coordinates in a cube
* @tparam A \ref layout_concept where each dimension of the cube specifies the corresponding stride.
*/
template <typename Shape_>
struct ComputeOffsetFromShape {
  static CUTLASS_HOST_DEVICE int get(int d, int h, int w, int c) {
    // clang-format off
    return d * Shape_::kH * Shape_::kW * Shape_::kC +
           h * Shape_::kW * Shape_::kC +
           w * Shape_::kC +
           c;
    // clang-format on
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief Compute the offset for the given coordinates in a cube
* @tparam A \ref layout_concept where each dimension of the cube specifies the corresponding stride.
*/
template <typename Strides_>
struct ComputeOffsetFromStrides {
  static CUTLASS_HOST_DEVICE int get(int d, int h, int w, int c) {
    return d * Strides_::kD + h * Strides_::kH + w * Strides_::kW + c * Strides_::kC;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief Decompose threadId.x into coordinate of a cube whose dimensions are specified by Threads_.
* Afterwards compute the offset of those coordinates using Strides_
* @tparam Threads_ The dimension of the cube the threadIdx.x value is mapped on
* @tparam Strides_ The strides to use when compute the offsets based on the coordinates of the cube.
*/
template <typename Threads_, typename Strides_>
struct ComputeThreadOffsetFromStrides {
  static CUTLASS_DEVICE int get() {
    // Decompose the thread index.
    int c = threadIdx.x % Threads_::kC;
    int w = threadIdx.x / Threads_::kC % Threads_::kW;
    int h = threadIdx.x / Threads_::kC / Threads_::kW % Threads_::kH;
    int d = threadIdx.x / Threads_::kC / Threads_::kW / Threads_::kH;

    // Compute the offset.
    return d * Strides_::kD + h * Strides_::kH + w * Strides_::kW + c * Strides_::kC;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
*@brief Specialization for D=1
*/
template <int T_h_, int T_w_, int T_c_, int S_h_, int S_w_, int S_c_>
struct ComputeThreadOffsetFromStrides<Shape<1, T_h_, T_w_, T_c_>, Shape<1, S_h_, S_w_, S_c_> > {
  static CUTLASS_DEVICE int get() {
    // Decompose the thread index.
    int c = threadIdx.x % T_c_;
    int w = threadIdx.x / T_c_ % T_w_;
    int h = threadIdx.x / T_c_ / T_w_ % T_h_;

    // Compute the offset.
    return h * S_h_ + w * S_w_ + c * S_c_;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
*@brief Specialization for D=1 and C=1
*/
template <int T_h_, int T_w_, int S_h_, int S_w_>
struct ComputeThreadOffsetFromStrides<Shape<1, T_h_, T_w_, 1>, Shape<1, S_h_, S_w_, 1> > {
  static CUTLASS_DEVICE int get() {
    // Decompose the thread index.
    int w = threadIdx.x % T_w_;
    int h = threadIdx.x / T_w_;

    // Compute the offset.
    return h * S_h_ + w * S_w_;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
