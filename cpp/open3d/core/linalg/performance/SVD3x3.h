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
/**************************************************************************
**
**  svd3
**
**  Quick singular value decomposition as described by:
**  A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis,
**  Computing the Singular Value Decomposition of 3x3 matrices
**  with minimal branching and elementary floating point operations,
**  University of Wisconsin - Madison technical report TR1690, May 2011
**
** 	Implementated by: Kui Wu
**	kwu@cs.utah.edu
**  Modified by: Wei Dong and Rishabh Singh
**
**  May 2018
**
**************************************************************************/

#pragma once

#include "open3d/core/CUDAUtils.h"

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
#include <cuda.h>
#endif

#include "math.h"
#include "open3d/core/linalg/performance/Matrix.h"

#define gone 1065353216
#define gsine_pi_over_eight 1053028117

#define gcosine_pi_over_eight 1064076127
#define gone_half 0.5f
#define gsmall_number 1.e-12f
#define gtiny_number 1.e-20f
#define gfour_gamma_squared 5.8284273147583007813f

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
#define __o3d__max(x, y) (x > y ? x : y)
#define __o3d__fadd_rn(x, y) __fadd_rn(x, y)
#define __o3d__fsub_rn(x, y) __fsub_rn(x, y)
#define __o3d__frsqrt_rn(x) __frsqrt_rn(x)

#define __o3d__dadd_rn(x, y) __dadd_rn(x, y)
#define __o3d__dsub_rn(x, y) __dsub_rn(x, y)
#define __o3d__drsqrt_rn(x) __drcp_rn(__dsqrt_rn(x))
#else

#define __o3d__max(x, y) (x > y ? x : y)
#define __o3d__fadd_rn(x, y) (x + y)
#define __o3d__fsub_rn(x, y) (x - y)
#define __o3d__frsqrt_rn(x) (1.0 / sqrt(x))

#define __o3d__dadd_rn(x, y) (x + y)
#define __o3d__dsub_rn(x, y) (x - y)
#define __o3d__drsqrt_rn(x) (1.0 / sqrt(x))

#define __o3d__add_rn(x, y) (x + y)
#define __o3d__sub_rn(x, y) (x - y)
#define __o3d__rsqrt_rn(x) (1.0 / sqrt(x))
#endif

template <typename scalar_t>
union un {
    scalar_t f;
    unsigned int ui;
};

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void svd3x3(const scalar_t a11,
                                              const scalar_t a12,
                                              const scalar_t a13,
                                              const scalar_t a21,
                                              const scalar_t a22,
                                              const scalar_t a23,
                                              const scalar_t a31,
                                              const scalar_t a32,
                                              const scalar_t a33,  // input A
                                              scalar_t &u11,
                                              scalar_t &u12,
                                              scalar_t &u13,
                                              scalar_t &u21,
                                              scalar_t &u22,
                                              scalar_t &u23,
                                              scalar_t &u31,
                                              scalar_t &u32,
                                              scalar_t &u33,  // output U
                                              scalar_t &s11,
                                              scalar_t &s22,
                                              scalar_t &s33,  // output S
                                              scalar_t &v11,
                                              scalar_t &v12,
                                              scalar_t &v13,
                                              scalar_t &v21,
                                              scalar_t &v22,
                                              scalar_t &v23,
                                              scalar_t &v31,
                                              scalar_t &v32,
                                              scalar_t &v33  // output V
);

template <>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void svd3x3<double>(
        const double a11,
        const double a12,
        const double a13,
        const double a21,
        const double a22,
        const double a23,
        const double a31,
        const double a32,
        const double a33,  // input A
        double &u11,
        double &u12,
        double &u13,
        double &u21,
        double &u22,
        double &u23,
        double &u31,
        double &u32,
        double &u33,  // output U
        double &s11,
        double &s22,
        double &s33,  // output S
        double &v11,
        double &v12,
        double &v13,
        double &v21,
        double &v22,
        double &v23,
        double &v31,
        double &v32,
        double &v33  // output V
) {
    un<double> Sa11, Sa21, Sa31, Sa12, Sa22, Sa32, Sa13, Sa23, Sa33;
    un<double> Su11, Su21, Su31, Su12, Su22, Su32, Su13, Su23, Su33;
    un<double> Sv11, Sv21, Sv31, Sv12, Sv22, Sv32, Sv13, Sv23, Sv33;
    un<double> Sc, Ss, Sch, Ssh;
    un<double> Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;
    un<double> Ss11, Ss21, Ss31, Ss22, Ss32, Ss33;
    un<double> Sqvs, Sqvvx, Sqvvy, Sqvvz;

    Sa11.f = a11;
    Sa12.f = a12;
    Sa13.f = a13;
    Sa21.f = a21;
    Sa22.f = a22;
    Sa23.f = a23;
    Sa31.f = a31;
    Sa32.f = a32;
    Sa33.f = a33;

    //###########################################################
    // Compute normal equations matrix
    //###########################################################

    Ss11.f = Sa11.f * Sa11.f;
    Stmp1.f = Sa21.f * Sa21.f;
    Ss11.f = __o3d__dadd_rn(Stmp1.f, Ss11.f);
    Stmp1.f = Sa31.f * Sa31.f;
    Ss11.f = __o3d__dadd_rn(Stmp1.f, Ss11.f);

    Ss21.f = Sa12.f * Sa11.f;
    Stmp1.f = Sa22.f * Sa21.f;
    Ss21.f = __o3d__dadd_rn(Stmp1.f, Ss21.f);
    Stmp1.f = Sa32.f * Sa31.f;
    Ss21.f = __o3d__dadd_rn(Stmp1.f, Ss21.f);

    Ss31.f = Sa13.f * Sa11.f;
    Stmp1.f = Sa23.f * Sa21.f;
    Ss31.f = __o3d__dadd_rn(Stmp1.f, Ss31.f);
    Stmp1.f = Sa33.f * Sa31.f;
    Ss31.f = __o3d__dadd_rn(Stmp1.f, Ss31.f);

    Ss22.f = Sa12.f * Sa12.f;
    Stmp1.f = Sa22.f * Sa22.f;
    Ss22.f = __o3d__dadd_rn(Stmp1.f, Ss22.f);
    Stmp1.f = Sa32.f * Sa32.f;
    Ss22.f = __o3d__dadd_rn(Stmp1.f, Ss22.f);

    Ss32.f = Sa13.f * Sa12.f;
    Stmp1.f = Sa23.f * Sa22.f;
    Ss32.f = __o3d__dadd_rn(Stmp1.f, Ss32.f);
    Stmp1.f = Sa33.f * Sa32.f;
    Ss32.f = __o3d__dadd_rn(Stmp1.f, Ss32.f);

    Ss33.f = Sa13.f * Sa13.f;
    Stmp1.f = Sa23.f * Sa23.f;
    Ss33.f = __o3d__dadd_rn(Stmp1.f, Ss33.f);
    Stmp1.f = Sa33.f * Sa33.f;
    Ss33.f = __o3d__dadd_rn(Stmp1.f, Ss33.f);

    Sqvs.f = 1.f;
    Sqvvx.f = 0.f;
    Sqvvy.f = 0.f;
    Sqvvz.f = 0.f;

    //###########################################################
    // Solve symmetric eigenproblem using Jacobi iteration
    //###########################################################
    for (int i = 0; i < 4; i++) {
        Ssh.f = Ss21.f * 0.5f;
        Stmp5.f = __o3d__dsub_rn(Ss11.f, Ss22.f);

        Stmp2.f = Ssh.f * Ssh.f;
        Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
        Ssh.ui = Stmp1.ui & Ssh.ui;
        Sch.ui = Stmp1.ui & Stmp5.ui;
        Stmp2.ui = ~Stmp1.ui & gone;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Stmp3.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
        Stmp4.f = __o3d__drsqrt_rn(Stmp3.f);

        Ssh.f = Stmp4.f * Ssh.f;
        Sch.f = Stmp4.f * Sch.f;
        Stmp1.f = gfour_gamma_squared * Stmp1.f;
        Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

        Stmp2.ui = gsine_pi_over_eight & Stmp1.ui;
        Ssh.ui = ~Stmp1.ui & Ssh.ui;
        Ssh.ui = Ssh.ui | Stmp2.ui;
        Stmp2.ui = gcosine_pi_over_eight & Stmp1.ui;
        Sch.ui = ~Stmp1.ui & Sch.ui;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Sc.f = __o3d__dsub_rn(Stmp2.f, Stmp1.f);
        Ss.f = Sch.f * Ssh.f;
        Ss.f = __o3d__dadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f,
               Sch.f);
#endif
        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        Stmp3.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
        Ss33.f = Ss33.f * Stmp3.f;
        Ss31.f = Ss31.f * Stmp3.f;
        Ss32.f = Ss32.f * Stmp3.f;
        Ss33.f = Ss33.f * Stmp3.f;

        Stmp1.f = Ss.f * Ss31.f;
        Stmp2.f = Ss.f * Ss32.f;
        Ss31.f = Sc.f * Ss31.f;
        Ss32.f = Sc.f * Ss32.f;
        Ss31.f = __o3d__dadd_rn(Stmp2.f, Ss31.f);
        Ss32.f = __o3d__dsub_rn(Ss32.f, Stmp1.f);

        Stmp2.f = Ss.f * Ss.f;
        Stmp1.f = Ss22.f * Stmp2.f;
        Stmp3.f = Ss11.f * Stmp2.f;
        Stmp4.f = Sc.f * Sc.f;
        Ss11.f = Ss11.f * Stmp4.f;
        Ss22.f = Ss22.f * Stmp4.f;
        Ss11.f = __o3d__dadd_rn(Ss11.f, Stmp1.f);
        Ss22.f = __o3d__dadd_rn(Ss22.f, Stmp3.f);
        Stmp4.f = __o3d__dsub_rn(Stmp4.f, Stmp2.f);
        Stmp2.f = __o3d__dadd_rn(Ss21.f, Ss21.f);
        Ss21.f = Ss21.f * Stmp4.f;
        Stmp4.f = Sc.f * Ss.f;
        Stmp2.f = Stmp2.f * Stmp4.f;
        Stmp5.f = Stmp5.f * Stmp4.f;
        Ss11.f = __o3d__dadd_rn(Ss11.f, Stmp2.f);
        Ss21.f = __o3d__dsub_rn(Ss21.f, Stmp5.f);
        Ss22.f = __o3d__dsub_rn(Ss22.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("%.20g\n", Ss11.f);
        printf("%.20g %.20g\n", Ss21.f, Ss22.f);
        printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        Stmp1.f = Ssh.f * Sqvvx.f;
        Stmp2.f = Ssh.f * Sqvvy.f;
        Stmp3.f = Ssh.f * Sqvvz.f;
        Ssh.f = Ssh.f * Sqvs.f;

        Sqvs.f = Sch.f * Sqvs.f;
        Sqvvx.f = Sch.f * Sqvvx.f;
        Sqvvy.f = Sch.f * Sqvvy.f;
        Sqvvz.f = Sch.f * Sqvvz.f;

        Sqvvz.f = __o3d__dadd_rn(Sqvvz.f, Ssh.f);
        Sqvs.f = __o3d__dsub_rn(Sqvs.f, Stmp3.f);
        Sqvvx.f = __o3d__dadd_rn(Sqvvx.f, Stmp2.f);
        Sqvvy.f = __o3d__dsub_rn(Sqvvy.f, Stmp1.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU q %.20g %.20g %.20g %.20g\n", Sqvvx.f, Sqvvy.f, Sqvvz.f,
               Sqvs.f);
#endif

        //////////////////////////////////////////////////////////////////////////
        // (1->3)
        //////////////////////////////////////////////////////////////////////////
        Ssh.f = Ss32.f * 0.5f;
        Stmp5.f = __o3d__dsub_rn(Ss22.f, Ss33.f);

        Stmp2.f = Ssh.f * Ssh.f;
        Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
        Ssh.ui = Stmp1.ui & Ssh.ui;
        Sch.ui = Stmp1.ui & Stmp5.ui;
        Stmp2.ui = ~Stmp1.ui & gone;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Stmp3.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
        Stmp4.f = __o3d__drsqrt_rn(Stmp3.f);

        Ssh.f = Stmp4.f * Ssh.f;
        Sch.f = Stmp4.f * Sch.f;
        Stmp1.f = gfour_gamma_squared * Stmp1.f;
        Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

        Stmp2.ui = gsine_pi_over_eight & Stmp1.ui;
        Ssh.ui = ~Stmp1.ui & Ssh.ui;
        Ssh.ui = Ssh.ui | Stmp2.ui;
        Stmp2.ui = gcosine_pi_over_eight & Stmp1.ui;
        Sch.ui = ~Stmp1.ui & Sch.ui;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Sc.f = __o3d__dsub_rn(Stmp2.f, Stmp1.f);
        Ss.f = Sch.f * Ssh.f;
        Ss.f = __o3d__dadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f,
               Sch.f);
#endif

        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        Stmp3.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
        Ss11.f = Ss11.f * Stmp3.f;
        Ss21.f = Ss21.f * Stmp3.f;
        Ss31.f = Ss31.f * Stmp3.f;
        Ss11.f = Ss11.f * Stmp3.f;

        Stmp1.f = Ss.f * Ss21.f;
        Stmp2.f = Ss.f * Ss31.f;
        Ss21.f = Sc.f * Ss21.f;
        Ss31.f = Sc.f * Ss31.f;
        Ss21.f = __o3d__dadd_rn(Stmp2.f, Ss21.f);
        Ss31.f = __o3d__dsub_rn(Ss31.f, Stmp1.f);

        Stmp2.f = Ss.f * Ss.f;
        Stmp1.f = Ss33.f * Stmp2.f;
        Stmp3.f = Ss22.f * Stmp2.f;
        Stmp4.f = Sc.f * Sc.f;
        Ss22.f = Ss22.f * Stmp4.f;
        Ss33.f = Ss33.f * Stmp4.f;
        Ss22.f = __o3d__dadd_rn(Ss22.f, Stmp1.f);
        Ss33.f = __o3d__dadd_rn(Ss33.f, Stmp3.f);
        Stmp4.f = __o3d__dsub_rn(Stmp4.f, Stmp2.f);
        Stmp2.f = __o3d__dadd_rn(Ss32.f, Ss32.f);
        Ss32.f = Ss32.f * Stmp4.f;
        Stmp4.f = Sc.f * Ss.f;
        Stmp2.f = Stmp2.f * Stmp4.f;
        Stmp5.f = Stmp5.f * Stmp4.f;
        Ss22.f = __o3d__dadd_rn(Ss22.f, Stmp2.f);
        Ss32.f = __o3d__dsub_rn(Ss32.f, Stmp5.f);
        Ss33.f = __o3d__dsub_rn(Ss33.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("%.20g\n", Ss11.f);
        printf("%.20g %.20g\n", Ss21.f, Ss22.f);
        printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        Stmp1.f = Ssh.f * Sqvvx.f;
        Stmp2.f = Ssh.f * Sqvvy.f;
        Stmp3.f = Ssh.f * Sqvvz.f;
        Ssh.f = Ssh.f * Sqvs.f;

        Sqvs.f = Sch.f * Sqvs.f;
        Sqvvx.f = Sch.f * Sqvvx.f;
        Sqvvy.f = Sch.f * Sqvvy.f;
        Sqvvz.f = Sch.f * Sqvvz.f;

        Sqvvx.f = __o3d__dadd_rn(Sqvvx.f, Ssh.f);
        Sqvs.f = __o3d__dsub_rn(Sqvs.f, Stmp1.f);
        Sqvvy.f = __o3d__dadd_rn(Sqvvy.f, Stmp3.f);
        Sqvvz.f = __o3d__dsub_rn(Sqvvz.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU q %.20g %.20g %.20g %.20g\n", Sqvvx.f, Sqvvy.f, Sqvvz.f,
               Sqvs.f);
#endif
#if 1
        //////////////////////////////////////////////////////////////////////////
        // 1 -> 2
        //////////////////////////////////////////////////////////////////////////

        Ssh.f = Ss31.f * 0.5f;
        Stmp5.f = __o3d__dsub_rn(Ss33.f, Ss11.f);

        Stmp2.f = Ssh.f * Ssh.f;
        Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
        Ssh.ui = Stmp1.ui & Ssh.ui;
        Sch.ui = Stmp1.ui & Stmp5.ui;
        Stmp2.ui = ~Stmp1.ui & gone;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Stmp3.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
        Stmp4.f = __o3d__drsqrt_rn(Stmp3.f);

        Ssh.f = Stmp4.f * Ssh.f;
        Sch.f = Stmp4.f * Sch.f;
        Stmp1.f = gfour_gamma_squared * Stmp1.f;
        Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

        Stmp2.ui = gsine_pi_over_eight & Stmp1.ui;
        Ssh.ui = ~Stmp1.ui & Ssh.ui;
        Ssh.ui = Ssh.ui | Stmp2.ui;
        Stmp2.ui = gcosine_pi_over_eight & Stmp1.ui;
        Sch.ui = ~Stmp1.ui & Sch.ui;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Sc.f = __o3d__dsub_rn(Stmp2.f, Stmp1.f);
        Ss.f = Sch.f * Ssh.f;
        Ss.f = __o3d__dadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f,
               Sch.f);
#endif

        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        Stmp3.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
        Ss22.f = Ss22.f * Stmp3.f;
        Ss32.f = Ss32.f * Stmp3.f;
        Ss21.f = Ss21.f * Stmp3.f;
        Ss22.f = Ss22.f * Stmp3.f;

        Stmp1.f = Ss.f * Ss32.f;
        Stmp2.f = Ss.f * Ss21.f;
        Ss32.f = Sc.f * Ss32.f;
        Ss21.f = Sc.f * Ss21.f;
        Ss32.f = __o3d__dadd_rn(Stmp2.f, Ss32.f);
        Ss21.f = __o3d__dsub_rn(Ss21.f, Stmp1.f);

        Stmp2.f = Ss.f * Ss.f;
        Stmp1.f = Ss11.f * Stmp2.f;
        Stmp3.f = Ss33.f * Stmp2.f;
        Stmp4.f = Sc.f * Sc.f;
        Ss33.f = Ss33.f * Stmp4.f;
        Ss11.f = Ss11.f * Stmp4.f;
        Ss33.f = __o3d__dadd_rn(Ss33.f, Stmp1.f);
        Ss11.f = __o3d__dadd_rn(Ss11.f, Stmp3.f);
        Stmp4.f = __o3d__dsub_rn(Stmp4.f, Stmp2.f);
        Stmp2.f = __o3d__dadd_rn(Ss31.f, Ss31.f);
        Ss31.f = Ss31.f * Stmp4.f;
        Stmp4.f = Sc.f * Ss.f;
        Stmp2.f = Stmp2.f * Stmp4.f;
        Stmp5.f = Stmp5.f * Stmp4.f;
        Ss33.f = __o3d__dadd_rn(Ss33.f, Stmp2.f);
        Ss31.f = __o3d__dsub_rn(Ss31.f, Stmp5.f);
        Ss11.f = __o3d__dsub_rn(Ss11.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("%.20g\n", Ss11.f);
        printf("%.20g %.20g\n", Ss21.f, Ss22.f);
        printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        Stmp1.f = Ssh.f * Sqvvx.f;
        Stmp2.f = Ssh.f * Sqvvy.f;
        Stmp3.f = Ssh.f * Sqvvz.f;
        Ssh.f = Ssh.f * Sqvs.f;

        Sqvs.f = Sch.f * Sqvs.f;
        Sqvvx.f = Sch.f * Sqvvx.f;
        Sqvvy.f = Sch.f * Sqvvy.f;
        Sqvvz.f = Sch.f * Sqvvz.f;

        Sqvvy.f = __o3d__dadd_rn(Sqvvy.f, Ssh.f);
        Sqvs.f = __o3d__dsub_rn(Sqvs.f, Stmp2.f);
        Sqvvz.f = __o3d__dadd_rn(Sqvvz.f, Stmp1.f);
        Sqvvx.f = __o3d__dsub_rn(Sqvvx.f, Stmp3.f);
#endif
    }

    //###########################################################
    // Normalize quaternion for matrix V
    //###########################################################

    Stmp2.f = Sqvs.f * Sqvs.f;
    Stmp1.f = Sqvvx.f * Sqvvx.f;
    Stmp2.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = Sqvvy.f * Sqvvy.f;
    Stmp2.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = Sqvvz.f * Sqvvz.f;
    Stmp2.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);

    Stmp1.f = __o3d__drsqrt_rn(Stmp2.f);
    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__dadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__dsub_rn(Stmp1.f, Stmp3.f);

    Sqvs.f = Sqvs.f * Stmp1.f;
    Sqvvx.f = Sqvvx.f * Stmp1.f;
    Sqvvy.f = Sqvvy.f * Stmp1.f;
    Sqvvz.f = Sqvvz.f * Stmp1.f;

    //###########################################################
    // Transform quaternion to matrix V
    //###########################################################

    Stmp1.f = Sqvvx.f * Sqvvx.f;
    Stmp2.f = Sqvvy.f * Sqvvy.f;
    Stmp3.f = Sqvvz.f * Sqvvz.f;
    Sv11.f = Sqvs.f * Sqvs.f;
    Sv22.f = __o3d__dsub_rn(Sv11.f, Stmp1.f);
    Sv33.f = __o3d__dsub_rn(Sv22.f, Stmp2.f);
    Sv33.f = __o3d__dadd_rn(Sv33.f, Stmp3.f);
    Sv22.f = __o3d__dadd_rn(Sv22.f, Stmp2.f);
    Sv22.f = __o3d__dsub_rn(Sv22.f, Stmp3.f);
    Sv11.f = __o3d__dadd_rn(Sv11.f, Stmp1.f);
    Sv11.f = __o3d__dsub_rn(Sv11.f, Stmp2.f);
    Sv11.f = __o3d__dsub_rn(Sv11.f, Stmp3.f);
    Stmp1.f = __o3d__dadd_rn(Sqvvx.f, Sqvvx.f);
    Stmp2.f = __o3d__dadd_rn(Sqvvy.f, Sqvvy.f);
    Stmp3.f = __o3d__dadd_rn(Sqvvz.f, Sqvvz.f);
    Sv32.f = Sqvs.f * Stmp1.f;
    Sv13.f = Sqvs.f * Stmp2.f;
    Sv21.f = Sqvs.f * Stmp3.f;
    Stmp1.f = Sqvvy.f * Stmp1.f;
    Stmp2.f = Sqvvz.f * Stmp2.f;
    Stmp3.f = Sqvvx.f * Stmp3.f;
    Sv12.f = __o3d__dsub_rn(Stmp1.f, Sv21.f);
    Sv23.f = __o3d__dsub_rn(Stmp2.f, Sv32.f);
    Sv31.f = __o3d__dsub_rn(Stmp3.f, Sv13.f);
    Sv21.f = __o3d__dadd_rn(Stmp1.f, Sv21.f);
    Sv32.f = __o3d__dadd_rn(Stmp2.f, Sv32.f);
    Sv13.f = __o3d__dadd_rn(Stmp3.f, Sv13.f);

    ///###########################################################
    // Multiply (from the right) with V
    //###########################################################

    Stmp2.f = Sa12.f;
    Stmp3.f = Sa13.f;
    Sa12.f = Sv12.f * Sa11.f;
    Sa13.f = Sv13.f * Sa11.f;
    Sa11.f = Sv11.f * Sa11.f;
    Stmp1.f = Sv21.f * Stmp2.f;
    Sa11.f = __o3d__dadd_rn(Sa11.f, Stmp1.f);
    Stmp1.f = Sv31.f * Stmp3.f;
    Sa11.f = __o3d__dadd_rn(Sa11.f, Stmp1.f);
    Stmp1.f = Sv22.f * Stmp2.f;
    Sa12.f = __o3d__dadd_rn(Sa12.f, Stmp1.f);
    Stmp1.f = Sv32.f * Stmp3.f;
    Sa12.f = __o3d__dadd_rn(Sa12.f, Stmp1.f);
    Stmp1.f = Sv23.f * Stmp2.f;
    Sa13.f = __o3d__dadd_rn(Sa13.f, Stmp1.f);
    Stmp1.f = Sv33.f * Stmp3.f;
    Sa13.f = __o3d__dadd_rn(Sa13.f, Stmp1.f);

    Stmp2.f = Sa22.f;
    Stmp3.f = Sa23.f;
    Sa22.f = Sv12.f * Sa21.f;
    Sa23.f = Sv13.f * Sa21.f;
    Sa21.f = Sv11.f * Sa21.f;
    Stmp1.f = Sv21.f * Stmp2.f;
    Sa21.f = __o3d__dadd_rn(Sa21.f, Stmp1.f);
    Stmp1.f = Sv31.f * Stmp3.f;
    Sa21.f = __o3d__dadd_rn(Sa21.f, Stmp1.f);
    Stmp1.f = Sv22.f * Stmp2.f;
    Sa22.f = __o3d__dadd_rn(Sa22.f, Stmp1.f);
    Stmp1.f = Sv32.f * Stmp3.f;
    Sa22.f = __o3d__dadd_rn(Sa22.f, Stmp1.f);
    Stmp1.f = Sv23.f * Stmp2.f;
    Sa23.f = __o3d__dadd_rn(Sa23.f, Stmp1.f);
    Stmp1.f = Sv33.f * Stmp3.f;
    Sa23.f = __o3d__dadd_rn(Sa23.f, Stmp1.f);

    Stmp2.f = Sa32.f;
    Stmp3.f = Sa33.f;
    Sa32.f = Sv12.f * Sa31.f;
    Sa33.f = Sv13.f * Sa31.f;
    Sa31.f = Sv11.f * Sa31.f;
    Stmp1.f = Sv21.f * Stmp2.f;
    Sa31.f = __o3d__dadd_rn(Sa31.f, Stmp1.f);
    Stmp1.f = Sv31.f * Stmp3.f;
    Sa31.f = __o3d__dadd_rn(Sa31.f, Stmp1.f);
    Stmp1.f = Sv22.f * Stmp2.f;
    Sa32.f = __o3d__dadd_rn(Sa32.f, Stmp1.f);
    Stmp1.f = Sv32.f * Stmp3.f;
    Sa32.f = __o3d__dadd_rn(Sa32.f, Stmp1.f);
    Stmp1.f = Sv23.f * Stmp2.f;
    Sa33.f = __o3d__dadd_rn(Sa33.f, Stmp1.f);
    Stmp1.f = Sv33.f * Stmp3.f;
    Sa33.f = __o3d__dadd_rn(Sa33.f, Stmp1.f);

    //###########################################################
    // Permute columns such that the singular values are sorted
    //###########################################################

    Stmp1.f = Sa11.f * Sa11.f;
    Stmp4.f = Sa21.f * Sa21.f;
    Stmp1.f = __o3d__dadd_rn(Stmp1.f, Stmp4.f);
    Stmp4.f = Sa31.f * Sa31.f;
    Stmp1.f = __o3d__dadd_rn(Stmp1.f, Stmp4.f);

    Stmp2.f = Sa12.f * Sa12.f;
    Stmp4.f = Sa22.f * Sa22.f;
    Stmp2.f = __o3d__dadd_rn(Stmp2.f, Stmp4.f);
    Stmp4.f = Sa32.f * Sa32.f;
    Stmp2.f = __o3d__dadd_rn(Stmp2.f, Stmp4.f);

    Stmp3.f = Sa13.f * Sa13.f;
    Stmp4.f = Sa23.f * Sa23.f;
    Stmp3.f = __o3d__dadd_rn(Stmp3.f, Stmp4.f);
    Stmp4.f = Sa33.f * Sa33.f;
    Stmp3.f = __o3d__dadd_rn(Stmp3.f, Stmp4.f);

    // Swap columns 1-2 if necessary

    Stmp4.ui = (Stmp1.f < Stmp2.f) ? 0xffffffff : 0;
    Stmp5.ui = Sa11.ui ^ Sa12.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa11.ui = Sa11.ui ^ Stmp5.ui;
    Sa12.ui = Sa12.ui ^ Stmp5.ui;

    Stmp5.ui = Sa21.ui ^ Sa22.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa21.ui = Sa21.ui ^ Stmp5.ui;
    Sa22.ui = Sa22.ui ^ Stmp5.ui;

    Stmp5.ui = Sa31.ui ^ Sa32.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa31.ui = Sa31.ui ^ Stmp5.ui;
    Sa32.ui = Sa32.ui ^ Stmp5.ui;

    Stmp5.ui = Sv11.ui ^ Sv12.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv11.ui = Sv11.ui ^ Stmp5.ui;
    Sv12.ui = Sv12.ui ^ Stmp5.ui;

    Stmp5.ui = Sv21.ui ^ Sv22.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv21.ui = Sv21.ui ^ Stmp5.ui;
    Sv22.ui = Sv22.ui ^ Stmp5.ui;

    Stmp5.ui = Sv31.ui ^ Sv32.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv31.ui = Sv31.ui ^ Stmp5.ui;
    Sv32.ui = Sv32.ui ^ Stmp5.ui;

    Stmp5.ui = Stmp1.ui ^ Stmp2.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp1.ui = Stmp1.ui ^ Stmp5.ui;
    Stmp2.ui = Stmp2.ui ^ Stmp5.ui;

    // If columns 1-2 have been swapped, negate 2nd column of A and V so that V
    // is still a rotation

    Stmp5.f = -2.f;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp4.f = 1.f;
    Stmp4.f = __o3d__dadd_rn(Stmp4.f, Stmp5.f);

    Sa12.f = Sa12.f * Stmp4.f;
    Sa22.f = Sa22.f * Stmp4.f;
    Sa32.f = Sa32.f * Stmp4.f;

    Sv12.f = Sv12.f * Stmp4.f;
    Sv22.f = Sv22.f * Stmp4.f;
    Sv32.f = Sv32.f * Stmp4.f;

    // Swap columns 1-3 if necessary

    Stmp4.ui = (Stmp1.f < Stmp3.f) ? 0xffffffff : 0;
    Stmp5.ui = Sa11.ui ^ Sa13.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa11.ui = Sa11.ui ^ Stmp5.ui;
    Sa13.ui = Sa13.ui ^ Stmp5.ui;

    Stmp5.ui = Sa21.ui ^ Sa23.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa21.ui = Sa21.ui ^ Stmp5.ui;
    Sa23.ui = Sa23.ui ^ Stmp5.ui;

    Stmp5.ui = Sa31.ui ^ Sa33.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa31.ui = Sa31.ui ^ Stmp5.ui;
    Sa33.ui = Sa33.ui ^ Stmp5.ui;

    Stmp5.ui = Sv11.ui ^ Sv13.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv11.ui = Sv11.ui ^ Stmp5.ui;
    Sv13.ui = Sv13.ui ^ Stmp5.ui;

    Stmp5.ui = Sv21.ui ^ Sv23.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv21.ui = Sv21.ui ^ Stmp5.ui;
    Sv23.ui = Sv23.ui ^ Stmp5.ui;

    Stmp5.ui = Sv31.ui ^ Sv33.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv31.ui = Sv31.ui ^ Stmp5.ui;
    Sv33.ui = Sv33.ui ^ Stmp5.ui;

    Stmp5.ui = Stmp1.ui ^ Stmp3.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp1.ui = Stmp1.ui ^ Stmp5.ui;
    Stmp3.ui = Stmp3.ui ^ Stmp5.ui;

    // If columns 1-3 have been swapped, negate 1st column of A and V so that V
    // is still a rotation

    Stmp5.f = -2.f;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp4.f = 1.f;
    Stmp4.f = __o3d__dadd_rn(Stmp4.f, Stmp5.f);

    Sa11.f = Sa11.f * Stmp4.f;
    Sa21.f = Sa21.f * Stmp4.f;
    Sa31.f = Sa31.f * Stmp4.f;

    Sv11.f = Sv11.f * Stmp4.f;
    Sv21.f = Sv21.f * Stmp4.f;
    Sv31.f = Sv31.f * Stmp4.f;

    // Swap columns 2-3 if necessary

    Stmp4.ui = (Stmp2.f < Stmp3.f) ? 0xffffffff : 0;
    Stmp5.ui = Sa12.ui ^ Sa13.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa12.ui = Sa12.ui ^ Stmp5.ui;
    Sa13.ui = Sa13.ui ^ Stmp5.ui;

    Stmp5.ui = Sa22.ui ^ Sa23.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa22.ui = Sa22.ui ^ Stmp5.ui;
    Sa23.ui = Sa23.ui ^ Stmp5.ui;

    Stmp5.ui = Sa32.ui ^ Sa33.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa32.ui = Sa32.ui ^ Stmp5.ui;
    Sa33.ui = Sa33.ui ^ Stmp5.ui;

    Stmp5.ui = Sv12.ui ^ Sv13.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv12.ui = Sv12.ui ^ Stmp5.ui;
    Sv13.ui = Sv13.ui ^ Stmp5.ui;

    Stmp5.ui = Sv22.ui ^ Sv23.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv22.ui = Sv22.ui ^ Stmp5.ui;
    Sv23.ui = Sv23.ui ^ Stmp5.ui;

    Stmp5.ui = Sv32.ui ^ Sv33.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv32.ui = Sv32.ui ^ Stmp5.ui;
    Sv33.ui = Sv33.ui ^ Stmp5.ui;

    Stmp5.ui = Stmp2.ui ^ Stmp3.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp2.ui = Stmp2.ui ^ Stmp5.ui;
    Stmp3.ui = Stmp3.ui ^ Stmp5.ui;

    // If columns 2-3 have been swapped, negate 3rd column of A and V so that V
    // is still a rotation

    Stmp5.f = -2.f;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp4.f = 1.f;
    Stmp4.f = __o3d__dadd_rn(Stmp4.f, Stmp5.f);

    Sa13.f = Sa13.f * Stmp4.f;
    Sa23.f = Sa23.f * Stmp4.f;
    Sa33.f = Sa33.f * Stmp4.f;

    Sv13.f = Sv13.f * Stmp4.f;
    Sv23.f = Sv23.f * Stmp4.f;
    Sv33.f = Sv33.f * Stmp4.f;

    //###########################################################
    // Construct QR factorization of A*V (=U*D) using Givens rotations
    //###########################################################

    Su11.f = 1.f;
    Su12.f = 0.f;
    Su13.f = 0.f;
    Su21.f = 0.f;
    Su22.f = 1.f;
    Su23.f = 0.f;
    Su31.f = 0.f;
    Su32.f = 0.f;
    Su33.f = 1.f;

    Ssh.f = Sa21.f * Sa21.f;
    Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
    Ssh.ui = Ssh.ui & Sa21.ui;

    Stmp5.f = 0.f;
    Sch.f = __o3d__dsub_rn(Stmp5.f, Sa11.f);
    Sch.f = __o3d__max(Sch.f, Sa11.f);
    Sch.f = __o3d__max(Sch.f, gsmall_number);
    Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__drsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__dadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__dsub_rn(Stmp1.f, Stmp3.f);
    Stmp1.f = Stmp1.f * Stmp2.f;

    Sch.f = __o3d__dadd_rn(Sch.f, Stmp1.f);

    Stmp1.ui = ~Stmp5.ui & Ssh.ui;
    Stmp2.ui = ~Stmp5.ui & Sch.ui;
    Sch.ui = Stmp5.ui & Sch.ui;
    Ssh.ui = Stmp5.ui & Ssh.ui;
    Sch.ui = Sch.ui | Stmp1.ui;
    Ssh.ui = Ssh.ui | Stmp2.ui;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__drsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__dadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__dsub_rn(Stmp1.f, Stmp3.f);

    Sch.f = Sch.f * Stmp1.f;
    Ssh.f = Ssh.f * Stmp1.f;

    Sc.f = Sch.f * Sch.f;
    Ss.f = Ssh.f * Ssh.f;
    Sc.f = __o3d__dsub_rn(Sc.f, Ss.f);
    Ss.f = Ssh.f * Sch.f;
    Ss.f = __o3d__dadd_rn(Ss.f, Ss.f);

    //###########################################################
    // Rotate matrix A
    //###########################################################

    Stmp1.f = Ss.f * Sa11.f;
    Stmp2.f = Ss.f * Sa21.f;
    Sa11.f = Sc.f * Sa11.f;
    Sa21.f = Sc.f * Sa21.f;
    Sa11.f = __o3d__dadd_rn(Sa11.f, Stmp2.f);
    Sa21.f = __o3d__dsub_rn(Sa21.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa12.f;
    Stmp2.f = Ss.f * Sa22.f;
    Sa12.f = Sc.f * Sa12.f;
    Sa22.f = Sc.f * Sa22.f;
    Sa12.f = __o3d__dadd_rn(Sa12.f, Stmp2.f);
    Sa22.f = __o3d__dsub_rn(Sa22.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa13.f;
    Stmp2.f = Ss.f * Sa23.f;
    Sa13.f = Sc.f * Sa13.f;
    Sa23.f = Sc.f * Sa23.f;
    Sa13.f = __o3d__dadd_rn(Sa13.f, Stmp2.f);
    Sa23.f = __o3d__dsub_rn(Sa23.f, Stmp1.f);

    //###########################################################
    // Update matrix U
    //###########################################################

    Stmp1.f = Ss.f * Su11.f;
    Stmp2.f = Ss.f * Su12.f;
    Su11.f = Sc.f * Su11.f;
    Su12.f = Sc.f * Su12.f;
    Su11.f = __o3d__dadd_rn(Su11.f, Stmp2.f);
    Su12.f = __o3d__dsub_rn(Su12.f, Stmp1.f);

    Stmp1.f = Ss.f * Su21.f;
    Stmp2.f = Ss.f * Su22.f;
    Su21.f = Sc.f * Su21.f;
    Su22.f = Sc.f * Su22.f;
    Su21.f = __o3d__dadd_rn(Su21.f, Stmp2.f);
    Su22.f = __o3d__dsub_rn(Su22.f, Stmp1.f);

    Stmp1.f = Ss.f * Su31.f;
    Stmp2.f = Ss.f * Su32.f;
    Su31.f = Sc.f * Su31.f;
    Su32.f = Sc.f * Su32.f;
    Su31.f = __o3d__dadd_rn(Su31.f, Stmp2.f);
    Su32.f = __o3d__dsub_rn(Su32.f, Stmp1.f);

    // Second Givens rotation

    Ssh.f = Sa31.f * Sa31.f;
    Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
    Ssh.ui = Ssh.ui & Sa31.ui;

    Stmp5.f = 0.f;
    Sch.f = __o3d__dsub_rn(Stmp5.f, Sa11.f);
    Sch.f = __o3d__max(Sch.f, Sa11.f);
    Sch.f = __o3d__max(Sch.f, gsmall_number);
    Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__drsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__dadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__dsub_rn(Stmp1.f, Stmp3.f);
    Stmp1.f = Stmp1.f * Stmp2.f;

    Sch.f = __o3d__dadd_rn(Sch.f, Stmp1.f);

    Stmp1.ui = ~Stmp5.ui & Ssh.ui;
    Stmp2.ui = ~Stmp5.ui & Sch.ui;
    Sch.ui = Stmp5.ui & Sch.ui;
    Ssh.ui = Stmp5.ui & Ssh.ui;
    Sch.ui = Sch.ui | Stmp1.ui;
    Ssh.ui = Ssh.ui | Stmp2.ui;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__drsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__dadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__dsub_rn(Stmp1.f, Stmp3.f);

    Sch.f = Sch.f * Stmp1.f;
    Ssh.f = Ssh.f * Stmp1.f;

    Sc.f = Sch.f * Sch.f;
    Ss.f = Ssh.f * Ssh.f;
    Sc.f = __o3d__dsub_rn(Sc.f, Ss.f);
    Ss.f = Ssh.f * Sch.f;
    Ss.f = __o3d__dadd_rn(Ss.f, Ss.f);

    //###########################################################
    // Rotate matrix A
    //###########################################################

    Stmp1.f = Ss.f * Sa11.f;
    Stmp2.f = Ss.f * Sa31.f;
    Sa11.f = Sc.f * Sa11.f;
    Sa31.f = Sc.f * Sa31.f;
    Sa11.f = __o3d__dadd_rn(Sa11.f, Stmp2.f);
    Sa31.f = __o3d__dsub_rn(Sa31.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa12.f;
    Stmp2.f = Ss.f * Sa32.f;
    Sa12.f = Sc.f * Sa12.f;
    Sa32.f = Sc.f * Sa32.f;
    Sa12.f = __o3d__dadd_rn(Sa12.f, Stmp2.f);
    Sa32.f = __o3d__dsub_rn(Sa32.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa13.f;
    Stmp2.f = Ss.f * Sa33.f;
    Sa13.f = Sc.f * Sa13.f;
    Sa33.f = Sc.f * Sa33.f;
    Sa13.f = __o3d__dadd_rn(Sa13.f, Stmp2.f);
    Sa33.f = __o3d__dsub_rn(Sa33.f, Stmp1.f);

    //###########################################################
    // Update matrix U
    //###########################################################

    Stmp1.f = Ss.f * Su11.f;
    Stmp2.f = Ss.f * Su13.f;
    Su11.f = Sc.f * Su11.f;
    Su13.f = Sc.f * Su13.f;
    Su11.f = __o3d__dadd_rn(Su11.f, Stmp2.f);
    Su13.f = __o3d__dsub_rn(Su13.f, Stmp1.f);

    Stmp1.f = Ss.f * Su21.f;
    Stmp2.f = Ss.f * Su23.f;
    Su21.f = Sc.f * Su21.f;
    Su23.f = Sc.f * Su23.f;
    Su21.f = __o3d__dadd_rn(Su21.f, Stmp2.f);
    Su23.f = __o3d__dsub_rn(Su23.f, Stmp1.f);

    Stmp1.f = Ss.f * Su31.f;
    Stmp2.f = Ss.f * Su33.f;
    Su31.f = Sc.f * Su31.f;
    Su33.f = Sc.f * Su33.f;
    Su31.f = __o3d__dadd_rn(Su31.f, Stmp2.f);
    Su33.f = __o3d__dsub_rn(Su33.f, Stmp1.f);

    // Third Givens Rotation

    Ssh.f = Sa32.f * Sa32.f;
    Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
    Ssh.ui = Ssh.ui & Sa32.ui;

    Stmp5.f = 0.f;
    Sch.f = __o3d__dsub_rn(Stmp5.f, Sa22.f);
    Sch.f = __o3d__max(Sch.f, Sa22.f);
    Sch.f = __o3d__max(Sch.f, gsmall_number);
    Stmp5.ui = (Sa22.f >= Stmp5.f) ? 0xffffffff : 0;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__drsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__dadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__dsub_rn(Stmp1.f, Stmp3.f);
    Stmp1.f = Stmp1.f * Stmp2.f;

    Sch.f = __o3d__dadd_rn(Sch.f, Stmp1.f);

    Stmp1.ui = ~Stmp5.ui & Ssh.ui;
    Stmp2.ui = ~Stmp5.ui & Sch.ui;
    Sch.ui = Stmp5.ui & Sch.ui;
    Ssh.ui = Stmp5.ui & Ssh.ui;
    Sch.ui = Sch.ui | Stmp1.ui;
    Ssh.ui = Ssh.ui | Stmp2.ui;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__dadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__drsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__dadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__dsub_rn(Stmp1.f, Stmp3.f);

    Sch.f = Sch.f * Stmp1.f;
    Ssh.f = Ssh.f * Stmp1.f;

    Sc.f = Sch.f * Sch.f;
    Ss.f = Ssh.f * Ssh.f;
    Sc.f = __o3d__dsub_rn(Sc.f, Ss.f);
    Ss.f = Ssh.f * Sch.f;
    Ss.f = __o3d__dadd_rn(Ss.f, Ss.f);

    //###########################################################
    // Rotate matrix A
    //###########################################################

    Stmp1.f = Ss.f * Sa21.f;
    Stmp2.f = Ss.f * Sa31.f;
    Sa21.f = Sc.f * Sa21.f;
    Sa31.f = Sc.f * Sa31.f;
    Sa21.f = __o3d__dadd_rn(Sa21.f, Stmp2.f);
    Sa31.f = __o3d__dsub_rn(Sa31.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa22.f;
    Stmp2.f = Ss.f * Sa32.f;
    Sa22.f = Sc.f * Sa22.f;
    Sa32.f = Sc.f * Sa32.f;
    Sa22.f = __o3d__dadd_rn(Sa22.f, Stmp2.f);
    Sa32.f = __o3d__dsub_rn(Sa32.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa23.f;
    Stmp2.f = Ss.f * Sa33.f;
    Sa23.f = Sc.f * Sa23.f;
    Sa33.f = Sc.f * Sa33.f;
    Sa23.f = __o3d__dadd_rn(Sa23.f, Stmp2.f);
    Sa33.f = __o3d__dsub_rn(Sa33.f, Stmp1.f);

    //###########################################################
    // Update matrix U
    //###########################################################

    Stmp1.f = Ss.f * Su12.f;
    Stmp2.f = Ss.f * Su13.f;
    Su12.f = Sc.f * Su12.f;
    Su13.f = Sc.f * Su13.f;
    Su12.f = __o3d__dadd_rn(Su12.f, Stmp2.f);
    Su13.f = __o3d__dsub_rn(Su13.f, Stmp1.f);

    Stmp1.f = Ss.f * Su22.f;
    Stmp2.f = Ss.f * Su23.f;
    Su22.f = Sc.f * Su22.f;
    Su23.f = Sc.f * Su23.f;
    Su22.f = __o3d__dadd_rn(Su22.f, Stmp2.f);
    Su23.f = __o3d__dsub_rn(Su23.f, Stmp1.f);

    Stmp1.f = Ss.f * Su32.f;
    Stmp2.f = Ss.f * Su33.f;
    Su32.f = Sc.f * Su32.f;
    Su33.f = Sc.f * Su33.f;
    Su32.f = __o3d__dadd_rn(Su32.f, Stmp2.f);
    Su33.f = __o3d__dsub_rn(Su33.f, Stmp1.f);

    v11 = Sv11.f;
    v12 = Sv12.f;
    v13 = Sv13.f;
    v21 = Sv21.f;
    v22 = Sv22.f;
    v23 = Sv23.f;
    v31 = Sv31.f;
    v32 = Sv32.f;
    v33 = Sv33.f;

    u11 = Su11.f;
    u12 = Su12.f;
    u13 = Su13.f;
    u21 = Su21.f;
    u22 = Su22.f;
    u23 = Su23.f;
    u31 = Su31.f;
    u32 = Su32.f;
    u33 = Su33.f;

    s11 = Sa11.f;
    // s12 = Sa12.f; s13 = Sa13.f; s21 = Sa21.f;
    s22 = Sa22.f;
    // s23 = Sa23.f; s31 = Sa31.f; s32 = Sa32.f;
    s33 = Sa33.f;
}

template <>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void svd3x3<float>(
        const float a11,
        const float a12,
        const float a13,
        const float a21,
        const float a22,
        const float a23,
        const float a31,
        const float a32,
        const float a33,  // input A
        float &u11,
        float &u12,
        float &u13,
        float &u21,
        float &u22,
        float &u23,
        float &u31,
        float &u32,
        float &u33,  // output U
        float &s11,
        float &s22,
        float &s33,  // output S
        float &v11,
        float &v12,
        float &v13,
        float &v21,
        float &v22,
        float &v23,
        float &v31,
        float &v32,
        float &v33  // output V
) {
    un<float> Sa11, Sa21, Sa31, Sa12, Sa22, Sa32, Sa13, Sa23, Sa33;
    un<float> Su11, Su21, Su31, Su12, Su22, Su32, Su13, Su23, Su33;
    un<float> Sv11, Sv21, Sv31, Sv12, Sv22, Sv32, Sv13, Sv23, Sv33;
    un<float> Sc, Ss, Sch, Ssh;
    un<float> Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;
    un<float> Ss11, Ss21, Ss31, Ss22, Ss32, Ss33;
    un<float> Sqvs, Sqvvx, Sqvvy, Sqvvz;

    Sa11.f = a11;
    Sa12.f = a12;
    Sa13.f = a13;
    Sa21.f = a21;
    Sa22.f = a22;
    Sa23.f = a23;
    Sa31.f = a31;
    Sa32.f = a32;
    Sa33.f = a33;

    //###########################################################
    // Compute normal equations matrix
    //###########################################################

    Ss11.f = Sa11.f * Sa11.f;
    Stmp1.f = Sa21.f * Sa21.f;
    Ss11.f = __o3d__fadd_rn(Stmp1.f, Ss11.f);
    Stmp1.f = Sa31.f * Sa31.f;
    Ss11.f = __o3d__fadd_rn(Stmp1.f, Ss11.f);

    Ss21.f = Sa12.f * Sa11.f;
    Stmp1.f = Sa22.f * Sa21.f;
    Ss21.f = __o3d__fadd_rn(Stmp1.f, Ss21.f);
    Stmp1.f = Sa32.f * Sa31.f;
    Ss21.f = __o3d__fadd_rn(Stmp1.f, Ss21.f);

    Ss31.f = Sa13.f * Sa11.f;
    Stmp1.f = Sa23.f * Sa21.f;
    Ss31.f = __o3d__fadd_rn(Stmp1.f, Ss31.f);
    Stmp1.f = Sa33.f * Sa31.f;
    Ss31.f = __o3d__fadd_rn(Stmp1.f, Ss31.f);

    Ss22.f = Sa12.f * Sa12.f;
    Stmp1.f = Sa22.f * Sa22.f;
    Ss22.f = __o3d__fadd_rn(Stmp1.f, Ss22.f);
    Stmp1.f = Sa32.f * Sa32.f;
    Ss22.f = __o3d__fadd_rn(Stmp1.f, Ss22.f);

    Ss32.f = Sa13.f * Sa12.f;
    Stmp1.f = Sa23.f * Sa22.f;
    Ss32.f = __o3d__fadd_rn(Stmp1.f, Ss32.f);
    Stmp1.f = Sa33.f * Sa32.f;
    Ss32.f = __o3d__fadd_rn(Stmp1.f, Ss32.f);

    Ss33.f = Sa13.f * Sa13.f;
    Stmp1.f = Sa23.f * Sa23.f;
    Ss33.f = __o3d__fadd_rn(Stmp1.f, Ss33.f);
    Stmp1.f = Sa33.f * Sa33.f;
    Ss33.f = __o3d__fadd_rn(Stmp1.f, Ss33.f);

    Sqvs.f = 1.f;
    Sqvvx.f = 0.f;
    Sqvvy.f = 0.f;
    Sqvvz.f = 0.f;

    //###########################################################
    // Solve symmetric eigenproblem using Jacobi iteration
    //###########################################################
    for (int i = 0; i < 4; i++) {
        Ssh.f = Ss21.f * 0.5f;
        Stmp5.f = __o3d__fsub_rn(Ss11.f, Ss22.f);

        Stmp2.f = Ssh.f * Ssh.f;
        Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
        Ssh.ui = Stmp1.ui & Ssh.ui;
        Sch.ui = Stmp1.ui & Stmp5.ui;
        Stmp2.ui = ~Stmp1.ui & gone;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Stmp3.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
        Stmp4.f = __o3d__frsqrt_rn(Stmp3.f);

        Ssh.f = Stmp4.f * Ssh.f;
        Sch.f = Stmp4.f * Sch.f;
        Stmp1.f = gfour_gamma_squared * Stmp1.f;
        Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

        Stmp2.ui = gsine_pi_over_eight & Stmp1.ui;
        Ssh.ui = ~Stmp1.ui & Ssh.ui;
        Ssh.ui = Ssh.ui | Stmp2.ui;
        Stmp2.ui = gcosine_pi_over_eight & Stmp1.ui;
        Sch.ui = ~Stmp1.ui & Sch.ui;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Sc.f = __o3d__fsub_rn(Stmp2.f, Stmp1.f);
        Ss.f = Sch.f * Ssh.f;
        Ss.f = __o3d__fadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f,
               Sch.f);
#endif
        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        Stmp3.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
        Ss33.f = Ss33.f * Stmp3.f;
        Ss31.f = Ss31.f * Stmp3.f;
        Ss32.f = Ss32.f * Stmp3.f;
        Ss33.f = Ss33.f * Stmp3.f;

        Stmp1.f = Ss.f * Ss31.f;
        Stmp2.f = Ss.f * Ss32.f;
        Ss31.f = Sc.f * Ss31.f;
        Ss32.f = Sc.f * Ss32.f;
        Ss31.f = __o3d__fadd_rn(Stmp2.f, Ss31.f);
        Ss32.f = __o3d__fsub_rn(Ss32.f, Stmp1.f);

        Stmp2.f = Ss.f * Ss.f;
        Stmp1.f = Ss22.f * Stmp2.f;
        Stmp3.f = Ss11.f * Stmp2.f;
        Stmp4.f = Sc.f * Sc.f;
        Ss11.f = Ss11.f * Stmp4.f;
        Ss22.f = Ss22.f * Stmp4.f;
        Ss11.f = __o3d__fadd_rn(Ss11.f, Stmp1.f);
        Ss22.f = __o3d__fadd_rn(Ss22.f, Stmp3.f);
        Stmp4.f = __o3d__fsub_rn(Stmp4.f, Stmp2.f);
        Stmp2.f = __o3d__fadd_rn(Ss21.f, Ss21.f);
        Ss21.f = Ss21.f * Stmp4.f;
        Stmp4.f = Sc.f * Ss.f;
        Stmp2.f = Stmp2.f * Stmp4.f;
        Stmp5.f = Stmp5.f * Stmp4.f;
        Ss11.f = __o3d__fadd_rn(Ss11.f, Stmp2.f);
        Ss21.f = __o3d__fsub_rn(Ss21.f, Stmp5.f);
        Ss22.f = __o3d__fsub_rn(Ss22.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("%.20g\n", Ss11.f);
        printf("%.20g %.20g\n", Ss21.f, Ss22.f);
        printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        Stmp1.f = Ssh.f * Sqvvx.f;
        Stmp2.f = Ssh.f * Sqvvy.f;
        Stmp3.f = Ssh.f * Sqvvz.f;
        Ssh.f = Ssh.f * Sqvs.f;

        Sqvs.f = Sch.f * Sqvs.f;
        Sqvvx.f = Sch.f * Sqvvx.f;
        Sqvvy.f = Sch.f * Sqvvy.f;
        Sqvvz.f = Sch.f * Sqvvz.f;

        Sqvvz.f = __o3d__fadd_rn(Sqvvz.f, Ssh.f);
        Sqvs.f = __o3d__fsub_rn(Sqvs.f, Stmp3.f);
        Sqvvx.f = __o3d__fadd_rn(Sqvvx.f, Stmp2.f);
        Sqvvy.f = __o3d__fsub_rn(Sqvvy.f, Stmp1.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU q %.20g %.20g %.20g %.20g\n", Sqvvx.f, Sqvvy.f, Sqvvz.f,
               Sqvs.f);
#endif

        //////////////////////////////////////////////////////////////////////////
        // (1->3)
        //////////////////////////////////////////////////////////////////////////
        Ssh.f = Ss32.f * 0.5f;
        Stmp5.f = __o3d__fsub_rn(Ss22.f, Ss33.f);

        Stmp2.f = Ssh.f * Ssh.f;
        Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
        Ssh.ui = Stmp1.ui & Ssh.ui;
        Sch.ui = Stmp1.ui & Stmp5.ui;
        Stmp2.ui = ~Stmp1.ui & gone;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Stmp3.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
        Stmp4.f = __o3d__frsqrt_rn(Stmp3.f);

        Ssh.f = Stmp4.f * Ssh.f;
        Sch.f = Stmp4.f * Sch.f;
        Stmp1.f = gfour_gamma_squared * Stmp1.f;
        Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

        Stmp2.ui = gsine_pi_over_eight & Stmp1.ui;
        Ssh.ui = ~Stmp1.ui & Ssh.ui;
        Ssh.ui = Ssh.ui | Stmp2.ui;
        Stmp2.ui = gcosine_pi_over_eight & Stmp1.ui;
        Sch.ui = ~Stmp1.ui & Sch.ui;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Sc.f = __o3d__fsub_rn(Stmp2.f, Stmp1.f);
        Ss.f = Sch.f * Ssh.f;
        Ss.f = __o3d__fadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f,
               Sch.f);
#endif

        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        Stmp3.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
        Ss11.f = Ss11.f * Stmp3.f;
        Ss21.f = Ss21.f * Stmp3.f;
        Ss31.f = Ss31.f * Stmp3.f;
        Ss11.f = Ss11.f * Stmp3.f;

        Stmp1.f = Ss.f * Ss21.f;
        Stmp2.f = Ss.f * Ss31.f;
        Ss21.f = Sc.f * Ss21.f;
        Ss31.f = Sc.f * Ss31.f;
        Ss21.f = __o3d__fadd_rn(Stmp2.f, Ss21.f);
        Ss31.f = __o3d__fsub_rn(Ss31.f, Stmp1.f);

        Stmp2.f = Ss.f * Ss.f;
        Stmp1.f = Ss33.f * Stmp2.f;
        Stmp3.f = Ss22.f * Stmp2.f;
        Stmp4.f = Sc.f * Sc.f;
        Ss22.f = Ss22.f * Stmp4.f;
        Ss33.f = Ss33.f * Stmp4.f;
        Ss22.f = __o3d__fadd_rn(Ss22.f, Stmp1.f);
        Ss33.f = __o3d__fadd_rn(Ss33.f, Stmp3.f);
        Stmp4.f = __o3d__fsub_rn(Stmp4.f, Stmp2.f);
        Stmp2.f = __o3d__fadd_rn(Ss32.f, Ss32.f);
        Ss32.f = Ss32.f * Stmp4.f;
        Stmp4.f = Sc.f * Ss.f;
        Stmp2.f = Stmp2.f * Stmp4.f;
        Stmp5.f = Stmp5.f * Stmp4.f;
        Ss22.f = __o3d__fadd_rn(Ss22.f, Stmp2.f);
        Ss32.f = __o3d__fsub_rn(Ss32.f, Stmp5.f);
        Ss33.f = __o3d__fsub_rn(Ss33.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("%.20g\n", Ss11.f);
        printf("%.20g %.20g\n", Ss21.f, Ss22.f);
        printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        Stmp1.f = Ssh.f * Sqvvx.f;
        Stmp2.f = Ssh.f * Sqvvy.f;
        Stmp3.f = Ssh.f * Sqvvz.f;
        Ssh.f = Ssh.f * Sqvs.f;

        Sqvs.f = Sch.f * Sqvs.f;
        Sqvvx.f = Sch.f * Sqvvx.f;
        Sqvvy.f = Sch.f * Sqvvy.f;
        Sqvvz.f = Sch.f * Sqvvz.f;

        Sqvvx.f = __o3d__fadd_rn(Sqvvx.f, Ssh.f);
        Sqvs.f = __o3d__fsub_rn(Sqvs.f, Stmp1.f);
        Sqvvy.f = __o3d__fadd_rn(Sqvvy.f, Stmp3.f);
        Sqvvz.f = __o3d__fsub_rn(Sqvvz.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU q %.20g %.20g %.20g %.20g\n", Sqvvx.f, Sqvvy.f, Sqvvz.f,
               Sqvs.f);
#endif
#if 1
        //////////////////////////////////////////////////////////////////////////
        // 1 -> 2
        //////////////////////////////////////////////////////////////////////////

        Ssh.f = Ss31.f * 0.5f;
        Stmp5.f = __o3d__fsub_rn(Ss33.f, Ss11.f);

        Stmp2.f = Ssh.f * Ssh.f;
        Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
        Ssh.ui = Stmp1.ui & Ssh.ui;
        Sch.ui = Stmp1.ui & Stmp5.ui;
        Stmp2.ui = ~Stmp1.ui & gone;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Stmp3.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
        Stmp4.f = __o3d__frsqrt_rn(Stmp3.f);

        Ssh.f = Stmp4.f * Ssh.f;
        Sch.f = Stmp4.f * Sch.f;
        Stmp1.f = gfour_gamma_squared * Stmp1.f;
        Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

        Stmp2.ui = gsine_pi_over_eight & Stmp1.ui;
        Ssh.ui = ~Stmp1.ui & Ssh.ui;
        Ssh.ui = Ssh.ui | Stmp2.ui;
        Stmp2.ui = gcosine_pi_over_eight & Stmp1.ui;
        Sch.ui = ~Stmp1.ui & Sch.ui;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Sc.f = __o3d__fsub_rn(Stmp2.f, Stmp1.f);
        Ss.f = Sch.f * Ssh.f;
        Ss.f = __o3d__fadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f,
               Sch.f);
#endif

        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        Stmp3.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
        Ss22.f = Ss22.f * Stmp3.f;
        Ss32.f = Ss32.f * Stmp3.f;
        Ss21.f = Ss21.f * Stmp3.f;
        Ss22.f = Ss22.f * Stmp3.f;

        Stmp1.f = Ss.f * Ss32.f;
        Stmp2.f = Ss.f * Ss21.f;
        Ss32.f = Sc.f * Ss32.f;
        Ss21.f = Sc.f * Ss21.f;
        Ss32.f = __o3d__fadd_rn(Stmp2.f, Ss32.f);
        Ss21.f = __o3d__fsub_rn(Ss21.f, Stmp1.f);

        Stmp2.f = Ss.f * Ss.f;
        Stmp1.f = Ss11.f * Stmp2.f;
        Stmp3.f = Ss33.f * Stmp2.f;
        Stmp4.f = Sc.f * Sc.f;
        Ss33.f = Ss33.f * Stmp4.f;
        Ss11.f = Ss11.f * Stmp4.f;
        Ss33.f = __o3d__fadd_rn(Ss33.f, Stmp1.f);
        Ss11.f = __o3d__fadd_rn(Ss11.f, Stmp3.f);
        Stmp4.f = __o3d__fsub_rn(Stmp4.f, Stmp2.f);
        Stmp2.f = __o3d__fadd_rn(Ss31.f, Ss31.f);
        Ss31.f = Ss31.f * Stmp4.f;
        Stmp4.f = Sc.f * Ss.f;
        Stmp2.f = Stmp2.f * Stmp4.f;
        Stmp5.f = Stmp5.f * Stmp4.f;
        Ss33.f = __o3d__fadd_rn(Ss33.f, Stmp2.f);
        Ss31.f = __o3d__fsub_rn(Ss31.f, Stmp5.f);
        Ss11.f = __o3d__fsub_rn(Ss11.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("%.20g\n", Ss11.f);
        printf("%.20g %.20g\n", Ss21.f, Ss22.f);
        printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        Stmp1.f = Ssh.f * Sqvvx.f;
        Stmp2.f = Ssh.f * Sqvvy.f;
        Stmp3.f = Ssh.f * Sqvvz.f;
        Ssh.f = Ssh.f * Sqvs.f;

        Sqvs.f = Sch.f * Sqvs.f;
        Sqvvx.f = Sch.f * Sqvvx.f;
        Sqvvy.f = Sch.f * Sqvvy.f;
        Sqvvz.f = Sch.f * Sqvvz.f;

        Sqvvy.f = __o3d__fadd_rn(Sqvvy.f, Ssh.f);
        Sqvs.f = __o3d__fsub_rn(Sqvs.f, Stmp2.f);
        Sqvvz.f = __o3d__fadd_rn(Sqvvz.f, Stmp1.f);
        Sqvvx.f = __o3d__fsub_rn(Sqvvx.f, Stmp3.f);
#endif
    }

    //###########################################################
    // Normalize quaternion for matrix V
    //###########################################################

    Stmp2.f = Sqvs.f * Sqvs.f;
    Stmp1.f = Sqvvx.f * Sqvvx.f;
    Stmp2.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = Sqvvy.f * Sqvvy.f;
    Stmp2.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = Sqvvz.f * Sqvvz.f;
    Stmp2.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);

    Stmp1.f = __o3d__frsqrt_rn(Stmp2.f);
    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__fadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__fsub_rn(Stmp1.f, Stmp3.f);

    Sqvs.f = Sqvs.f * Stmp1.f;
    Sqvvx.f = Sqvvx.f * Stmp1.f;
    Sqvvy.f = Sqvvy.f * Stmp1.f;
    Sqvvz.f = Sqvvz.f * Stmp1.f;

    //###########################################################
    // Transform quaternion to matrix V
    //###########################################################

    Stmp1.f = Sqvvx.f * Sqvvx.f;
    Stmp2.f = Sqvvy.f * Sqvvy.f;
    Stmp3.f = Sqvvz.f * Sqvvz.f;
    Sv11.f = Sqvs.f * Sqvs.f;
    Sv22.f = __o3d__fsub_rn(Sv11.f, Stmp1.f);
    Sv33.f = __o3d__fsub_rn(Sv22.f, Stmp2.f);
    Sv33.f = __o3d__fadd_rn(Sv33.f, Stmp3.f);
    Sv22.f = __o3d__fadd_rn(Sv22.f, Stmp2.f);
    Sv22.f = __o3d__fsub_rn(Sv22.f, Stmp3.f);
    Sv11.f = __o3d__fadd_rn(Sv11.f, Stmp1.f);
    Sv11.f = __o3d__fsub_rn(Sv11.f, Stmp2.f);
    Sv11.f = __o3d__fsub_rn(Sv11.f, Stmp3.f);
    Stmp1.f = __o3d__fadd_rn(Sqvvx.f, Sqvvx.f);
    Stmp2.f = __o3d__fadd_rn(Sqvvy.f, Sqvvy.f);
    Stmp3.f = __o3d__fadd_rn(Sqvvz.f, Sqvvz.f);
    Sv32.f = Sqvs.f * Stmp1.f;
    Sv13.f = Sqvs.f * Stmp2.f;
    Sv21.f = Sqvs.f * Stmp3.f;
    Stmp1.f = Sqvvy.f * Stmp1.f;
    Stmp2.f = Sqvvz.f * Stmp2.f;
    Stmp3.f = Sqvvx.f * Stmp3.f;
    Sv12.f = __o3d__fsub_rn(Stmp1.f, Sv21.f);
    Sv23.f = __o3d__fsub_rn(Stmp2.f, Sv32.f);
    Sv31.f = __o3d__fsub_rn(Stmp3.f, Sv13.f);
    Sv21.f = __o3d__fadd_rn(Stmp1.f, Sv21.f);
    Sv32.f = __o3d__fadd_rn(Stmp2.f, Sv32.f);
    Sv13.f = __o3d__fadd_rn(Stmp3.f, Sv13.f);

    ///###########################################################
    // Multiply (from the right) with V
    //###########################################################

    Stmp2.f = Sa12.f;
    Stmp3.f = Sa13.f;
    Sa12.f = Sv12.f * Sa11.f;
    Sa13.f = Sv13.f * Sa11.f;
    Sa11.f = Sv11.f * Sa11.f;
    Stmp1.f = Sv21.f * Stmp2.f;
    Sa11.f = __o3d__fadd_rn(Sa11.f, Stmp1.f);
    Stmp1.f = Sv31.f * Stmp3.f;
    Sa11.f = __o3d__fadd_rn(Sa11.f, Stmp1.f);
    Stmp1.f = Sv22.f * Stmp2.f;
    Sa12.f = __o3d__fadd_rn(Sa12.f, Stmp1.f);
    Stmp1.f = Sv32.f * Stmp3.f;
    Sa12.f = __o3d__fadd_rn(Sa12.f, Stmp1.f);
    Stmp1.f = Sv23.f * Stmp2.f;
    Sa13.f = __o3d__fadd_rn(Sa13.f, Stmp1.f);
    Stmp1.f = Sv33.f * Stmp3.f;
    Sa13.f = __o3d__fadd_rn(Sa13.f, Stmp1.f);

    Stmp2.f = Sa22.f;
    Stmp3.f = Sa23.f;
    Sa22.f = Sv12.f * Sa21.f;
    Sa23.f = Sv13.f * Sa21.f;
    Sa21.f = Sv11.f * Sa21.f;
    Stmp1.f = Sv21.f * Stmp2.f;
    Sa21.f = __o3d__fadd_rn(Sa21.f, Stmp1.f);
    Stmp1.f = Sv31.f * Stmp3.f;
    Sa21.f = __o3d__fadd_rn(Sa21.f, Stmp1.f);
    Stmp1.f = Sv22.f * Stmp2.f;
    Sa22.f = __o3d__fadd_rn(Sa22.f, Stmp1.f);
    Stmp1.f = Sv32.f * Stmp3.f;
    Sa22.f = __o3d__fadd_rn(Sa22.f, Stmp1.f);
    Stmp1.f = Sv23.f * Stmp2.f;
    Sa23.f = __o3d__fadd_rn(Sa23.f, Stmp1.f);
    Stmp1.f = Sv33.f * Stmp3.f;
    Sa23.f = __o3d__fadd_rn(Sa23.f, Stmp1.f);

    Stmp2.f = Sa32.f;
    Stmp3.f = Sa33.f;
    Sa32.f = Sv12.f * Sa31.f;
    Sa33.f = Sv13.f * Sa31.f;
    Sa31.f = Sv11.f * Sa31.f;
    Stmp1.f = Sv21.f * Stmp2.f;
    Sa31.f = __o3d__fadd_rn(Sa31.f, Stmp1.f);
    Stmp1.f = Sv31.f * Stmp3.f;
    Sa31.f = __o3d__fadd_rn(Sa31.f, Stmp1.f);
    Stmp1.f = Sv22.f * Stmp2.f;
    Sa32.f = __o3d__fadd_rn(Sa32.f, Stmp1.f);
    Stmp1.f = Sv32.f * Stmp3.f;
    Sa32.f = __o3d__fadd_rn(Sa32.f, Stmp1.f);
    Stmp1.f = Sv23.f * Stmp2.f;
    Sa33.f = __o3d__fadd_rn(Sa33.f, Stmp1.f);
    Stmp1.f = Sv33.f * Stmp3.f;
    Sa33.f = __o3d__fadd_rn(Sa33.f, Stmp1.f);

    //###########################################################
    // Permute columns such that the singular values are sorted
    //###########################################################

    Stmp1.f = Sa11.f * Sa11.f;
    Stmp4.f = Sa21.f * Sa21.f;
    Stmp1.f = __o3d__fadd_rn(Stmp1.f, Stmp4.f);
    Stmp4.f = Sa31.f * Sa31.f;
    Stmp1.f = __o3d__fadd_rn(Stmp1.f, Stmp4.f);

    Stmp2.f = Sa12.f * Sa12.f;
    Stmp4.f = Sa22.f * Sa22.f;
    Stmp2.f = __o3d__fadd_rn(Stmp2.f, Stmp4.f);
    Stmp4.f = Sa32.f * Sa32.f;
    Stmp2.f = __o3d__fadd_rn(Stmp2.f, Stmp4.f);

    Stmp3.f = Sa13.f * Sa13.f;
    Stmp4.f = Sa23.f * Sa23.f;
    Stmp3.f = __o3d__fadd_rn(Stmp3.f, Stmp4.f);
    Stmp4.f = Sa33.f * Sa33.f;
    Stmp3.f = __o3d__fadd_rn(Stmp3.f, Stmp4.f);

    // Swap columns 1-2 if necessary

    Stmp4.ui = (Stmp1.f < Stmp2.f) ? 0xffffffff : 0;
    Stmp5.ui = Sa11.ui ^ Sa12.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa11.ui = Sa11.ui ^ Stmp5.ui;
    Sa12.ui = Sa12.ui ^ Stmp5.ui;

    Stmp5.ui = Sa21.ui ^ Sa22.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa21.ui = Sa21.ui ^ Stmp5.ui;
    Sa22.ui = Sa22.ui ^ Stmp5.ui;

    Stmp5.ui = Sa31.ui ^ Sa32.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa31.ui = Sa31.ui ^ Stmp5.ui;
    Sa32.ui = Sa32.ui ^ Stmp5.ui;

    Stmp5.ui = Sv11.ui ^ Sv12.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv11.ui = Sv11.ui ^ Stmp5.ui;
    Sv12.ui = Sv12.ui ^ Stmp5.ui;

    Stmp5.ui = Sv21.ui ^ Sv22.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv21.ui = Sv21.ui ^ Stmp5.ui;
    Sv22.ui = Sv22.ui ^ Stmp5.ui;

    Stmp5.ui = Sv31.ui ^ Sv32.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv31.ui = Sv31.ui ^ Stmp5.ui;
    Sv32.ui = Sv32.ui ^ Stmp5.ui;

    Stmp5.ui = Stmp1.ui ^ Stmp2.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp1.ui = Stmp1.ui ^ Stmp5.ui;
    Stmp2.ui = Stmp2.ui ^ Stmp5.ui;

    // If columns 1-2 have been swapped, negate 2nd column of A and V so that V
    // is still a rotation

    Stmp5.f = -2.f;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp4.f = 1.f;
    Stmp4.f = __o3d__fadd_rn(Stmp4.f, Stmp5.f);

    Sa12.f = Sa12.f * Stmp4.f;
    Sa22.f = Sa22.f * Stmp4.f;
    Sa32.f = Sa32.f * Stmp4.f;

    Sv12.f = Sv12.f * Stmp4.f;
    Sv22.f = Sv22.f * Stmp4.f;
    Sv32.f = Sv32.f * Stmp4.f;

    // Swap columns 1-3 if necessary

    Stmp4.ui = (Stmp1.f < Stmp3.f) ? 0xffffffff : 0;
    Stmp5.ui = Sa11.ui ^ Sa13.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa11.ui = Sa11.ui ^ Stmp5.ui;
    Sa13.ui = Sa13.ui ^ Stmp5.ui;

    Stmp5.ui = Sa21.ui ^ Sa23.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa21.ui = Sa21.ui ^ Stmp5.ui;
    Sa23.ui = Sa23.ui ^ Stmp5.ui;

    Stmp5.ui = Sa31.ui ^ Sa33.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa31.ui = Sa31.ui ^ Stmp5.ui;
    Sa33.ui = Sa33.ui ^ Stmp5.ui;

    Stmp5.ui = Sv11.ui ^ Sv13.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv11.ui = Sv11.ui ^ Stmp5.ui;
    Sv13.ui = Sv13.ui ^ Stmp5.ui;

    Stmp5.ui = Sv21.ui ^ Sv23.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv21.ui = Sv21.ui ^ Stmp5.ui;
    Sv23.ui = Sv23.ui ^ Stmp5.ui;

    Stmp5.ui = Sv31.ui ^ Sv33.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv31.ui = Sv31.ui ^ Stmp5.ui;
    Sv33.ui = Sv33.ui ^ Stmp5.ui;

    Stmp5.ui = Stmp1.ui ^ Stmp3.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp1.ui = Stmp1.ui ^ Stmp5.ui;
    Stmp3.ui = Stmp3.ui ^ Stmp5.ui;

    // If columns 1-3 have been swapped, negate 1st column of A and V so that V
    // is still a rotation

    Stmp5.f = -2.f;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp4.f = 1.f;
    Stmp4.f = __o3d__fadd_rn(Stmp4.f, Stmp5.f);

    Sa11.f = Sa11.f * Stmp4.f;
    Sa21.f = Sa21.f * Stmp4.f;
    Sa31.f = Sa31.f * Stmp4.f;

    Sv11.f = Sv11.f * Stmp4.f;
    Sv21.f = Sv21.f * Stmp4.f;
    Sv31.f = Sv31.f * Stmp4.f;

    // Swap columns 2-3 if necessary

    Stmp4.ui = (Stmp2.f < Stmp3.f) ? 0xffffffff : 0;
    Stmp5.ui = Sa12.ui ^ Sa13.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa12.ui = Sa12.ui ^ Stmp5.ui;
    Sa13.ui = Sa13.ui ^ Stmp5.ui;

    Stmp5.ui = Sa22.ui ^ Sa23.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa22.ui = Sa22.ui ^ Stmp5.ui;
    Sa23.ui = Sa23.ui ^ Stmp5.ui;

    Stmp5.ui = Sa32.ui ^ Sa33.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sa32.ui = Sa32.ui ^ Stmp5.ui;
    Sa33.ui = Sa33.ui ^ Stmp5.ui;

    Stmp5.ui = Sv12.ui ^ Sv13.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv12.ui = Sv12.ui ^ Stmp5.ui;
    Sv13.ui = Sv13.ui ^ Stmp5.ui;

    Stmp5.ui = Sv22.ui ^ Sv23.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv22.ui = Sv22.ui ^ Stmp5.ui;
    Sv23.ui = Sv23.ui ^ Stmp5.ui;

    Stmp5.ui = Sv32.ui ^ Sv33.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Sv32.ui = Sv32.ui ^ Stmp5.ui;
    Sv33.ui = Sv33.ui ^ Stmp5.ui;

    Stmp5.ui = Stmp2.ui ^ Stmp3.ui;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp2.ui = Stmp2.ui ^ Stmp5.ui;
    Stmp3.ui = Stmp3.ui ^ Stmp5.ui;

    // If columns 2-3 have been swapped, negate 3rd column of A and V so that V
    // is still a rotation

    Stmp5.f = -2.f;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp4.f = 1.f;
    Stmp4.f = __o3d__fadd_rn(Stmp4.f, Stmp5.f);

    Sa13.f = Sa13.f * Stmp4.f;
    Sa23.f = Sa23.f * Stmp4.f;
    Sa33.f = Sa33.f * Stmp4.f;

    Sv13.f = Sv13.f * Stmp4.f;
    Sv23.f = Sv23.f * Stmp4.f;
    Sv33.f = Sv33.f * Stmp4.f;

    //###########################################################
    // Construct QR factorization of A*V (=U*D) using Givens rotations
    //###########################################################

    Su11.f = 1.f;
    Su12.f = 0.f;
    Su13.f = 0.f;
    Su21.f = 0.f;
    Su22.f = 1.f;
    Su23.f = 0.f;
    Su31.f = 0.f;
    Su32.f = 0.f;
    Su33.f = 1.f;

    Ssh.f = Sa21.f * Sa21.f;
    Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
    Ssh.ui = Ssh.ui & Sa21.ui;

    Stmp5.f = 0.f;
    Sch.f = __o3d__fsub_rn(Stmp5.f, Sa11.f);
    Sch.f = __o3d__max(Sch.f, Sa11.f);
    Sch.f = __o3d__max(Sch.f, gsmall_number);
    Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__frsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__fadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__fsub_rn(Stmp1.f, Stmp3.f);
    Stmp1.f = Stmp1.f * Stmp2.f;

    Sch.f = __o3d__fadd_rn(Sch.f, Stmp1.f);

    Stmp1.ui = ~Stmp5.ui & Ssh.ui;
    Stmp2.ui = ~Stmp5.ui & Sch.ui;
    Sch.ui = Stmp5.ui & Sch.ui;
    Ssh.ui = Stmp5.ui & Ssh.ui;
    Sch.ui = Sch.ui | Stmp1.ui;
    Ssh.ui = Ssh.ui | Stmp2.ui;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__frsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__fadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__fsub_rn(Stmp1.f, Stmp3.f);

    Sch.f = Sch.f * Stmp1.f;
    Ssh.f = Ssh.f * Stmp1.f;

    Sc.f = Sch.f * Sch.f;
    Ss.f = Ssh.f * Ssh.f;
    Sc.f = __o3d__fsub_rn(Sc.f, Ss.f);
    Ss.f = Ssh.f * Sch.f;
    Ss.f = __o3d__fadd_rn(Ss.f, Ss.f);

    //###########################################################
    // Rotate matrix A
    //###########################################################

    Stmp1.f = Ss.f * Sa11.f;
    Stmp2.f = Ss.f * Sa21.f;
    Sa11.f = Sc.f * Sa11.f;
    Sa21.f = Sc.f * Sa21.f;
    Sa11.f = __o3d__fadd_rn(Sa11.f, Stmp2.f);
    Sa21.f = __o3d__fsub_rn(Sa21.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa12.f;
    Stmp2.f = Ss.f * Sa22.f;
    Sa12.f = Sc.f * Sa12.f;
    Sa22.f = Sc.f * Sa22.f;
    Sa12.f = __o3d__fadd_rn(Sa12.f, Stmp2.f);
    Sa22.f = __o3d__fsub_rn(Sa22.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa13.f;
    Stmp2.f = Ss.f * Sa23.f;
    Sa13.f = Sc.f * Sa13.f;
    Sa23.f = Sc.f * Sa23.f;
    Sa13.f = __o3d__fadd_rn(Sa13.f, Stmp2.f);
    Sa23.f = __o3d__fsub_rn(Sa23.f, Stmp1.f);

    //###########################################################
    // Update matrix U
    //###########################################################

    Stmp1.f = Ss.f * Su11.f;
    Stmp2.f = Ss.f * Su12.f;
    Su11.f = Sc.f * Su11.f;
    Su12.f = Sc.f * Su12.f;
    Su11.f = __o3d__fadd_rn(Su11.f, Stmp2.f);
    Su12.f = __o3d__fsub_rn(Su12.f, Stmp1.f);

    Stmp1.f = Ss.f * Su21.f;
    Stmp2.f = Ss.f * Su22.f;
    Su21.f = Sc.f * Su21.f;
    Su22.f = Sc.f * Su22.f;
    Su21.f = __o3d__fadd_rn(Su21.f, Stmp2.f);
    Su22.f = __o3d__fsub_rn(Su22.f, Stmp1.f);

    Stmp1.f = Ss.f * Su31.f;
    Stmp2.f = Ss.f * Su32.f;
    Su31.f = Sc.f * Su31.f;
    Su32.f = Sc.f * Su32.f;
    Su31.f = __o3d__fadd_rn(Su31.f, Stmp2.f);
    Su32.f = __o3d__fsub_rn(Su32.f, Stmp1.f);

    // Second Givens rotation

    Ssh.f = Sa31.f * Sa31.f;
    Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
    Ssh.ui = Ssh.ui & Sa31.ui;

    Stmp5.f = 0.f;
    Sch.f = __o3d__fsub_rn(Stmp5.f, Sa11.f);
    Sch.f = __o3d__max(Sch.f, Sa11.f);
    Sch.f = __o3d__max(Sch.f, gsmall_number);
    Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__frsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__fadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__fsub_rn(Stmp1.f, Stmp3.f);
    Stmp1.f = Stmp1.f * Stmp2.f;

    Sch.f = __o3d__fadd_rn(Sch.f, Stmp1.f);

    Stmp1.ui = ~Stmp5.ui & Ssh.ui;
    Stmp2.ui = ~Stmp5.ui & Sch.ui;
    Sch.ui = Stmp5.ui & Sch.ui;
    Ssh.ui = Stmp5.ui & Ssh.ui;
    Sch.ui = Sch.ui | Stmp1.ui;
    Ssh.ui = Ssh.ui | Stmp2.ui;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__frsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__fadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__fsub_rn(Stmp1.f, Stmp3.f);

    Sch.f = Sch.f * Stmp1.f;
    Ssh.f = Ssh.f * Stmp1.f;

    Sc.f = Sch.f * Sch.f;
    Ss.f = Ssh.f * Ssh.f;
    Sc.f = __o3d__fsub_rn(Sc.f, Ss.f);
    Ss.f = Ssh.f * Sch.f;
    Ss.f = __o3d__fadd_rn(Ss.f, Ss.f);

    //###########################################################
    // Rotate matrix A
    //###########################################################

    Stmp1.f = Ss.f * Sa11.f;
    Stmp2.f = Ss.f * Sa31.f;
    Sa11.f = Sc.f * Sa11.f;
    Sa31.f = Sc.f * Sa31.f;
    Sa11.f = __o3d__fadd_rn(Sa11.f, Stmp2.f);
    Sa31.f = __o3d__fsub_rn(Sa31.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa12.f;
    Stmp2.f = Ss.f * Sa32.f;
    Sa12.f = Sc.f * Sa12.f;
    Sa32.f = Sc.f * Sa32.f;
    Sa12.f = __o3d__fadd_rn(Sa12.f, Stmp2.f);
    Sa32.f = __o3d__fsub_rn(Sa32.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa13.f;
    Stmp2.f = Ss.f * Sa33.f;
    Sa13.f = Sc.f * Sa13.f;
    Sa33.f = Sc.f * Sa33.f;
    Sa13.f = __o3d__fadd_rn(Sa13.f, Stmp2.f);
    Sa33.f = __o3d__fsub_rn(Sa33.f, Stmp1.f);

    //###########################################################
    // Update matrix U
    //###########################################################

    Stmp1.f = Ss.f * Su11.f;
    Stmp2.f = Ss.f * Su13.f;
    Su11.f = Sc.f * Su11.f;
    Su13.f = Sc.f * Su13.f;
    Su11.f = __o3d__fadd_rn(Su11.f, Stmp2.f);
    Su13.f = __o3d__fsub_rn(Su13.f, Stmp1.f);

    Stmp1.f = Ss.f * Su21.f;
    Stmp2.f = Ss.f * Su23.f;
    Su21.f = Sc.f * Su21.f;
    Su23.f = Sc.f * Su23.f;
    Su21.f = __o3d__fadd_rn(Su21.f, Stmp2.f);
    Su23.f = __o3d__fsub_rn(Su23.f, Stmp1.f);

    Stmp1.f = Ss.f * Su31.f;
    Stmp2.f = Ss.f * Su33.f;
    Su31.f = Sc.f * Su31.f;
    Su33.f = Sc.f * Su33.f;
    Su31.f = __o3d__fadd_rn(Su31.f, Stmp2.f);
    Su33.f = __o3d__fsub_rn(Su33.f, Stmp1.f);

    // Third Givens Rotation

    Ssh.f = Sa32.f * Sa32.f;
    Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
    Ssh.ui = Ssh.ui & Sa32.ui;

    Stmp5.f = 0.f;
    Sch.f = __o3d__fsub_rn(Stmp5.f, Sa22.f);
    Sch.f = __o3d__max(Sch.f, Sa22.f);
    Sch.f = __o3d__max(Sch.f, gsmall_number);
    Stmp5.ui = (Sa22.f >= Stmp5.f) ? 0xffffffff : 0;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__frsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__fadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__fsub_rn(Stmp1.f, Stmp3.f);
    Stmp1.f = Stmp1.f * Stmp2.f;

    Sch.f = __o3d__fadd_rn(Sch.f, Stmp1.f);

    Stmp1.ui = ~Stmp5.ui & Ssh.ui;
    Stmp2.ui = ~Stmp5.ui & Sch.ui;
    Sch.ui = Stmp5.ui & Sch.ui;
    Ssh.ui = Stmp5.ui & Ssh.ui;
    Sch.ui = Sch.ui | Stmp1.ui;
    Ssh.ui = Ssh.ui | Stmp2.ui;

    Stmp1.f = Sch.f * Sch.f;
    Stmp2.f = Ssh.f * Ssh.f;
    Stmp2.f = __o3d__fadd_rn(Stmp1.f, Stmp2.f);
    Stmp1.f = __o3d__frsqrt_rn(Stmp2.f);

    Stmp4.f = Stmp1.f * 0.5f;
    Stmp3.f = Stmp1.f * Stmp4.f;
    Stmp3.f = Stmp1.f * Stmp3.f;
    Stmp3.f = Stmp2.f * Stmp3.f;
    Stmp1.f = __o3d__fadd_rn(Stmp1.f, Stmp4.f);
    Stmp1.f = __o3d__fsub_rn(Stmp1.f, Stmp3.f);

    Sch.f = Sch.f * Stmp1.f;
    Ssh.f = Ssh.f * Stmp1.f;

    Sc.f = Sch.f * Sch.f;
    Ss.f = Ssh.f * Ssh.f;
    Sc.f = __o3d__fsub_rn(Sc.f, Ss.f);
    Ss.f = Ssh.f * Sch.f;
    Ss.f = __o3d__fadd_rn(Ss.f, Ss.f);

    //###########################################################
    // Rotate matrix A
    //###########################################################

    Stmp1.f = Ss.f * Sa21.f;
    Stmp2.f = Ss.f * Sa31.f;
    Sa21.f = Sc.f * Sa21.f;
    Sa31.f = Sc.f * Sa31.f;
    Sa21.f = __o3d__fadd_rn(Sa21.f, Stmp2.f);
    Sa31.f = __o3d__fsub_rn(Sa31.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa22.f;
    Stmp2.f = Ss.f * Sa32.f;
    Sa22.f = Sc.f * Sa22.f;
    Sa32.f = Sc.f * Sa32.f;
    Sa22.f = __o3d__fadd_rn(Sa22.f, Stmp2.f);
    Sa32.f = __o3d__fsub_rn(Sa32.f, Stmp1.f);

    Stmp1.f = Ss.f * Sa23.f;
    Stmp2.f = Ss.f * Sa33.f;
    Sa23.f = Sc.f * Sa23.f;
    Sa33.f = Sc.f * Sa33.f;
    Sa23.f = __o3d__fadd_rn(Sa23.f, Stmp2.f);
    Sa33.f = __o3d__fsub_rn(Sa33.f, Stmp1.f);

    //###########################################################
    // Update matrix U
    //###########################################################

    Stmp1.f = Ss.f * Su12.f;
    Stmp2.f = Ss.f * Su13.f;
    Su12.f = Sc.f * Su12.f;
    Su13.f = Sc.f * Su13.f;
    Su12.f = __o3d__fadd_rn(Su12.f, Stmp2.f);
    Su13.f = __o3d__fsub_rn(Su13.f, Stmp1.f);

    Stmp1.f = Ss.f * Su22.f;
    Stmp2.f = Ss.f * Su23.f;
    Su22.f = Sc.f * Su22.f;
    Su23.f = Sc.f * Su23.f;
    Su22.f = __o3d__fadd_rn(Su22.f, Stmp2.f);
    Su23.f = __o3d__fsub_rn(Su23.f, Stmp1.f);

    Stmp1.f = Ss.f * Su32.f;
    Stmp2.f = Ss.f * Su33.f;
    Su32.f = Sc.f * Su32.f;
    Su33.f = Sc.f * Su33.f;
    Su32.f = __o3d__fadd_rn(Su32.f, Stmp2.f);
    Su33.f = __o3d__fsub_rn(Su33.f, Stmp1.f);

    v11 = Sv11.f;
    v12 = Sv12.f;
    v13 = Sv13.f;
    v21 = Sv21.f;
    v22 = Sv22.f;
    v23 = Sv23.f;
    v31 = Sv31.f;
    v32 = Sv32.f;
    v33 = Sv33.f;

    u11 = Su11.f;
    u12 = Su12.f;
    u13 = Su13.f;
    u21 = Su21.f;
    u22 = Su22.f;
    u23 = Su23.f;
    u31 = Su31.f;
    u32 = Su32.f;
    u33 = Su33.f;

    s11 = Sa11.f;
    // s12 = Sa12.f; s13 = Sa13.f; s21 = Sa21.f;
    s22 = Sa22.f;
    // s23 = Sa23.f; s31 = Sa31.f; s32 = Sa32.f;
    s33 = Sa33.f;
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void svd3x3(const scalar_t *A_3x3,
                                              scalar_t *U_3x3,
                                              scalar_t *S_3x1,
                                              scalar_t *V_3x3) {
    svd3x3(A_3x3[0], A_3x3[1], A_3x3[2], A_3x3[3], A_3x3[4], A_3x3[5], A_3x3[6],
           A_3x3[7], A_3x3[8], U_3x3[0], U_3x3[1], U_3x3[2], U_3x3[3], U_3x3[4],
           U_3x3[5], U_3x3[6], U_3x3[7], U_3x3[8], S_3x1[0], S_3x1[1], S_3x1[2],
           V_3x3[0], V_3x3[1], V_3x3[2], V_3x3[3], V_3x3[4], V_3x3[5], V_3x3[6],
           V_3x3[7], V_3x3[8]);
    return;
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void solve_svd3x3(
        const scalar_t &a11,
        const scalar_t &a12,
        const scalar_t &a13,
        const scalar_t &a21,
        const scalar_t &a22,
        const scalar_t &a23,
        const scalar_t &a31,
        const scalar_t &a32,
        const scalar_t &a33,  // input A {3,3}
        const scalar_t &b1,
        const scalar_t &b2,
        const scalar_t &b3,  // input b {3,1}
        scalar_t &x1,
        scalar_t &x2,
        scalar_t &x3)  // output x {3,1}
{
    scalar_t U[9];
    scalar_t V[9];
    scalar_t S[3];
    svd3x3(a11, a12, a13, a21, a22, a23, a31, a32, a33, U[0], U[1], U[2], U[3],
           U[4], U[5], U[6], U[7], U[8], S[0], S[1], S[2], V[0], V[1], V[2],
           V[3], V[4], V[5], V[6], V[7], V[8]);

    //###########################################################
    // Sigma^+
    //###########################################################
    const scalar_t epsilon = 1e-6;
    S[0] = S[0] < epsilon ? 0 : 1.0 / S[0];
    S[1] = S[1] < epsilon ? 0 : 1.0 / S[1];
    S[2] = S[2] < epsilon ? 0 : 1.0 / S[2];

    //###########################################################
    // Ainv = V * [(Sigma^+) * UT]
    //###########################################################
    scalar_t Ainv[9] = {0};
    matmul3x3_3x3(V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8],
                  U[0] * S[0], U[3] * S[0], U[6] * S[0], U[1] * S[1],
                  U[4] * S[1], U[7] * S[1], U[2] * S[2], U[5] * S[2],
                  U[8] * S[2], Ainv[0], Ainv[1], Ainv[2], Ainv[3], Ainv[4],
                  Ainv[5], Ainv[6], Ainv[7], Ainv[8]);

    //###########################################################
    // x = Ainv * b
    //###########################################################
    matmul3x3_3x1(Ainv[0], Ainv[1], Ainv[2], Ainv[3], Ainv[4], Ainv[5], Ainv[6],
                  Ainv[7], Ainv[8], b1, b2, b3, x1, x2, x3);
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void solve_svd3x3(
        const scalar_t *A_3x3,  // input A {3,3}
        const scalar_t *B_3x1,  // input b {3,1}
        scalar_t *X_3x1)        // output x {3,1}
{
    solve_svd3x3(A_3x3[0], A_3x3[1], A_3x3[2], A_3x3[3], A_3x3[4], A_3x3[5],
                 A_3x3[6], A_3x3[7], A_3x3[8], B_3x1[0], B_3x1[1], B_3x1[2],
                 X_3x1[0], X_3x1[1], X_3x1[2]);
    return;
}
