/***************************************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
    \brief Basic copy routines for tensor views
*/

#pragma once

#include "cutlass/fragment.h"
#include "cutlass/layout/thread/tensor_foreach.h"
#include "cutlass/tensor_view.h"

namespace cutlass {
namespace layout {
namespace thread {

/// Define a functor that performs a copy operation on a tensor.
template <typename View_dst, typename View_src>
struct CopyFunc {
  /// Coordinate of index space
  typedef typename View_dst::TensorCoord TensorCoord;

  View_dst dst;

  View_src src;

  /// Constructor
  CUTLASS_DEVICE
  CopyFunc(View_dst dst, View_src src) : dst(dst), src(src) {}

  /// copy function
  CUTLASS_DEVICE
  void operator()(TensorCoord const& coord) {
    dst.at(coord) = src.at(coord);  // uses tensor view's map()
  }
};

template <typename T_dst, typename T_src, int rank, typename MapFunc_dst, typename MapFunc_src>
struct Copy {
  CUTLASS_DEVICE void copy(cutlass::TensorView<T_dst, rank, MapFunc_dst> dst,
                           cutlass::TensorView<T_src, rank, MapFunc_src> src) {
    // Define a functor called by TensorForEach<>
    typedef CopyFunc<cutlass::TensorView<T_dst, rank, MapFunc_dst>,
                     cutlass::TensorView<T_src, rank, MapFunc_src> >
        CopyFunc;

    // Instantiate on device with TensorViews
    CopyFunc copy_func(dst, src);

    // Invoke device-side for-each computation on the tensor
    cutlass::layout::thread::TensorForEach<CopyFunc,
                                           rank,  // View::kRank
                                           CopyFunc>(src.size(), copy_func);
  }
};

#if !defined(__CUDACC_RTC__) || defined(CUTLASS_NVRTC_HAS_FP16)
template <int rank>
struct Copy<half, half, rank, cutlass::MatrixLayout::RowMajor, cutlass::MatrixLayout::RowMajor> {
  CUTLASS_DEVICE void copy(cutlass::TensorView<half, rank, cutlass::MatrixLayout::RowMajor> dst,
                           cutlass::TensorView<half, rank, cutlass::MatrixLayout::RowMajor> src) {
    bool isPacked = dst.isPacked() && src.isPacked();
    if (isPacked) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < src.capacity(); ++i) {
        dst.at(i) = src.at(i);
      }
    } else {
      typedef CopyFunc<cutlass::TensorView<half, rank, cutlass::MatrixLayout::RowMajor>,
                       cutlass::TensorView<half, rank, cutlass::MatrixLayout::RowMajor> >
          CopyFunc;

      // Instantiate on device with TensorViews
      CopyFunc copy_func(dst, src);

      // Invoke device-side for-each computation on the tensor
      cutlass::layout::thread::TensorForEach<CopyFunc,
                                             rank,  // View::kRank
                                             CopyFunc>(src.size(), copy_func);
    }
  }
};

/// hgemm swizzle
/// Transform a fragment.
template <>
struct Copy<half, half, 2, cutlass::MatrixLayout::RowMajor, cutlass::MatrixLayout::ColumnMajor> {
  CUTLASS_DEVICE void copy(
      cutlass::TensorView<half, 2, cutlass::MatrixLayout::RowMajor> dst,
      cutlass::TensorView<half, 2, cutlass::MatrixLayout::ColumnMajor> src) {
    // Expose src/dst as int arrays.
    int const* src_int = reinterpret_cast<int const*>(src.const_ref().data());
    int* dst_int = reinterpret_cast<int*>(dst.ref().data());

    int kD = src.size(0);
    int kDhw = src.size(0) * src.size(1);

    // Transpose the data.
    // CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < kD; ++d) {
      // The indices to read two consecutive "rows".
      int const i0 = 2 * d + 0;
      int const i1 = 2 * d + 1;

      int a0 = src_int[i0];
      int a1 = src_int[i1];

      int b0, b1;
      asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b0) : "r"(a0), "r"(a1));
      asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(b1) : "r"(a0), "r"(a1));

      // The indices to store with "strides".
      int const j0 = 0 * (kDhw / 2) + d;
      int const j1 = 1 * (kDhw / 2) + d;

      dst_int[j0] = b0;
      dst_int[j1] = b1;
    }
  }
};
#endif

/// igemm swizzle
/// Transform a fragment.
template <>
struct Copy<int8_t,
            int8_t,
            2,
            cutlass::MatrixLayout::RowMajor,
            cutlass::MatrixLayout::ColumnMajor> {
  CUTLASS_DEVICE void copy(
      cutlass::TensorView<int8_t, 2, cutlass::MatrixLayout::RowMajor> dst,
      cutlass::TensorView<int8_t, 2, cutlass::MatrixLayout::ColumnMajor> src) {
    // Expose src/dst as int arrays.
    int const* src_int = reinterpret_cast<int const*>(src.const_ref().data());
    int* dst_int = reinterpret_cast<int*>(dst.ref().data());

    int kD = src.size(0);
    int kH = src.size(1);
    int kWc = src.stride(0);
    int kHwc = kH * kWc;

    // Transpose the data.
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < kH / 4; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < kWc / 4; ++w) {
          int const i0 = d * (kHwc / 4) + (4 * h + 0) * (kWc / 4) + w;
          int const i1 = d * (kHwc / 4) + (4 * h + 1) * (kWc / 4) + w;
          int const i2 = d * (kHwc / 4) + (4 * h + 2) * (kWc / 4) + w;
          int const i3 = d * (kHwc / 4) + (4 * h + 3) * (kWc / 4) + w;

          int a0 = src_int[i0];
          int a1 = src_int[i1];
          int a2 = src_int[i2];
          int a3 = src_int[i3];

          int b0, b1, b2, b3, c0;
          asm volatile("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(b0) : "r"(a0), "r"(a1));
          asm volatile("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(c0) : "r"(a2), "r"(a3));
          asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b0) : "r"(b0), "r"(c0));

          asm volatile("prmt.b32 %0, %1, %2, 0x0051;" : "=r"(b1) : "r"(a0), "r"(a1));
          asm volatile("prmt.b32 %0, %1, %2, 0x0051;" : "=r"(c0) : "r"(a2), "r"(a3));
          asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b1) : "r"(b1), "r"(c0));

          asm volatile("prmt.b32 %0, %1, %2, 0x0062;" : "=r"(b2) : "r"(a0), "r"(a1));
          asm volatile("prmt.b32 %0, %1, %2, 0x0062;" : "=r"(c0) : "r"(a2), "r"(a3));
          asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b2) : "r"(b2), "r"(c0));

          asm volatile("prmt.b32 %0, %1, %2, 0x0073;" : "=r"(b3) : "r"(a0), "r"(a1));
          asm volatile("prmt.b32 %0, %1, %2, 0x0073;" : "=r"(c0) : "r"(a2), "r"(a3));
          asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b3) : "r"(b3), "r"(c0));

          dst_int[i0] = b0;
          dst_int[i1] = b1;
          dst_int[i2] = b2;
          dst_int[i3] = b3;
        }
      }
    }
  }
};

template <typename Shape,
          int Rank,
          typename DstType,
          typename DstLayout,
          typename SrcType,
          typename SrcLayout>
struct Transform {

  typedef Fragment<DstType, ShapeCount<Shape>::kCount> DstFragment;
  typedef Fragment<SrcType, ShapeCount<Shape>::kCount> SrcFragment;

  /// The input fragment.
  typedef SrcFragment InputFragment;
  /// The output fragment.
  typedef DstFragment OutputFragment;

  CUTLASS_DEVICE void transform(SrcFragment& src, DstFragment& dst) {
    cutlass::TensorView<DstType, Rank, DstLayout> dstView(
        &dst[0],                            // pointer to base of matrix in device memory
        cutlass::make_Coord(Shape::kD, 1),  // stride vector
        cutlass::make_Coord(Shape::kD,
                            Shape::kH * Shape::kW)  // bounds of matrix
        );
    cutlass::TensorView<SrcType, Rank, SrcLayout> srcView(
        &src[0],                            // pointer to base of matrix in device memory
        cutlass::make_Coord(Shape::kD, 1),  // stride vector
        cutlass::make_Coord(Shape::kD,
                            Shape::kH * Shape::kW)  // bounds of matrix
        );
    cutlass::layout::thread::Copy<DstType, SrcType, Rank, DstLayout, SrcLayout> Transformer;
    Transformer.copy(dstView, srcView);
  }
};

#if !defined(__CUDACC_RTC__) || defined(CUTLASS_NVRTC_HAS_FP16)
template <typename Shape, int Rank, typename DstLayout, typename SrcLayout>
struct Transform<Shape, Rank, half, DstLayout, half, SrcLayout> {
  typedef Fragment<half, ShapeCount<Shape>::kCount> DstFragment;
  typedef Fragment<half, ShapeCount<Shape>::kCount> SrcFragment;

  /// The input fragment.
  typedef SrcFragment InputFragment;
  /// The output fragment.
  typedef DstFragment OutputFragment;

  CUTLASS_DEVICE void transform(SrcFragment& src, DstFragment& dst) {
    cutlass::TensorView<half, Rank, DstLayout> dstView(
        &dst[0],                            // pointer to base of matrix in device memory
        cutlass::make_Coord(Shape::kD, 1),  // stride vector
        cutlass::make_Coord(Shape::kD,
                            Shape::kH * Shape::kW)  // bounds of matrix
        );
    cutlass::TensorView<half, Rank, SrcLayout> srcView(
        &src[0],                            // pointer to base of matrix in device memory
        cutlass::make_Coord(Shape::kD, 1),  // stride vector
        cutlass::make_Coord(Shape::kD,
                            Shape::kH * Shape::kW)  // bounds of matrix
        );
    cutlass::layout::thread::Copy<half, half, Rank, DstLayout, SrcLayout> Transformer;
    Transformer.copy(dstView, srcView);
  }
};
#endif

template <typename Shape, int Rank, typename DstLayout, typename SrcLayout>
struct Transform<Shape, Rank, int8_t, DstLayout, int8_t, SrcLayout> {
  typedef Fragment<int8_t, ShapeCount<Shape>::kCount> DstFragment;
  typedef Fragment<int8_t, ShapeCount<Shape>::kCount> SrcFragment;

  /// The input fragment.
  typedef SrcFragment InputFragment;
  /// The output fragment.
  typedef DstFragment OutputFragment;

  CUTLASS_DEVICE void transform(SrcFragment& src, DstFragment& dst) {
    cutlass::TensorView<int8_t, Rank, DstLayout> dstView(
        &dst[0],  // pointer to base of matrix in device memory
        cutlass::make_Coord(Shape::kW * Shape::kC, 1),  // stride vector
        cutlass::make_Coord(Shape::kD,
                            Shape::kH)  // bounds of matrix
        );
    cutlass::TensorView<int8_t, Rank, SrcLayout> srcView(
        &src[0],  // pointer to base of matrix in device memory
        cutlass::make_Coord(Shape::kW * Shape::kC, 1),  // stride vector
        cutlass::make_Coord(Shape::kD,
                            Shape::kH)  // bounds of matrix
        );
    cutlass::layout::thread::Copy<int8_t, int8_t, Rank, DstLayout, SrcLayout> Transformer;
    Transformer.copy(dstView, srcView);
  }
};

}  // namespace thread
}  // namespace layout
}  // namespace cutlass
