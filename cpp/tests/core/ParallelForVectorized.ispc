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

#include "open3d/core/ParallelForVectorized.isph"

static inline void Lambda1(int64 idx, int64* uniform v) { v[idx] = idx; }
static inline void Lambda2(int64 idx, int64* uniform v, bool* uniform arg2) {
    Lambda1(idx, v);
    *arg2 = true;
}
static inline void Lambda3(int64 idx,
                           int64* uniform v,
                           bool* uniform arg2,
                           uint8* uniform arg3) {
    Lambda2(idx, v, arg2);
    *arg3 = 3;
}
static inline void Lambda4(int64 idx,
                           int64* uniform v,
                           bool* uniform arg2,
                           uint8* uniform arg3,
                           int8* uniform arg4) {
    Lambda3(idx, v, arg2, arg3);
    *arg4 = 4;
}
static inline void Lambda5(int64 idx,
                           int64* uniform v,
                           bool* uniform arg2,
                           uint8* uniform arg3,
                           int8* uniform arg4,
                           uint16* uniform arg5) {
    Lambda4(idx, v, arg2, arg3, arg4);
    *arg5 = 5;
}
static inline void Lambda6(int64 idx,
                           int64* uniform v,
                           bool* uniform arg2,
                           uint8* uniform arg3,
                           int8* uniform arg4,
                           uint16* uniform arg5,
                           int16* uniform arg6) {
    Lambda5(idx, v, arg2, arg3, arg4, arg5);
    *arg6 = 6;
}
static inline void Lambda7(int64 idx,
                           int64* uniform v,
                           bool* uniform arg2,
                           uint8* uniform arg3,
                           int8* uniform arg4,
                           uint16* uniform arg5,
                           int16* uniform arg6,
                           uint32* uniform arg7) {
    Lambda6(idx, v, arg2, arg3, arg4, arg5, arg6);
    *arg7 = 7;
}
static inline void Lambda8(int64 idx,
                           int64* uniform v,
                           bool* uniform arg2,
                           uint8* uniform arg3,
                           int8* uniform arg4,
                           uint16* uniform arg5,
                           int16* uniform arg6,
                           uint32* uniform arg7,
                           int32* uniform arg8) {
    Lambda7(idx, v, arg2, arg3, arg4, arg5, arg6, arg7);
    *arg8 = 8;
}

static inline void Lambda9(int64 idx,
                           int64* uniform v,
                           bool* uniform arg2,
                           uint8* uniform arg3,
                           int8* uniform arg4,
                           uint16* uniform arg5,
                           int16* uniform arg6,
                           uint32* uniform arg7,
                           int32* uniform arg8,
                           uint64* uniform arg9) {
    Lambda8(idx, v, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    *arg9 = 9;
}

OPEN3D_EXPORT_VECTORIZED_LAMBDA(LambdaKernel1, Lambda1, int64* uniform)
OPEN3D_EXPORT_VECTORIZED_LAMBDA(LambdaKernel2,
                                Lambda2,
                                int64* uniform,
                                bool* uniform)
OPEN3D_EXPORT_VECTORIZED_LAMBDA(
        LambdaKernel3, Lambda3, int64* uniform, bool* uniform, uint8* uniform)
OPEN3D_EXPORT_VECTORIZED_LAMBDA(LambdaKernel4,
                                Lambda4,
                                int64* uniform,
                                bool* uniform,
                                uint8* uniform,
                                int8* uniform)
OPEN3D_EXPORT_VECTORIZED_LAMBDA(LambdaKernel5,
                                Lambda5,
                                int64* uniform,
                                bool* uniform,
                                uint8* uniform,
                                int8* uniform,
                                uint16* uniform)
OPEN3D_EXPORT_VECTORIZED_LAMBDA(LambdaKernel6,
                                Lambda6,
                                int64* uniform,
                                bool* uniform,
                                uint8* uniform,
                                int8* uniform,
                                uint16* uniform,
                                int16* uniform)
OPEN3D_EXPORT_VECTORIZED_LAMBDA(LambdaKernel7,
                                Lambda7,
                                int64* uniform,
                                bool* uniform,
                                uint8* uniform,
                                int8* uniform,
                                uint16* uniform,
                                int16* uniform,
                                uint32* uniform)
OPEN3D_EXPORT_VECTORIZED_LAMBDA(LambdaKernel8,
                                Lambda8,
                                int64* uniform,
                                bool* uniform,
                                uint8* uniform,
                                int8* uniform,
                                uint16* uniform,
                                int16* uniform,
                                uint32* uniform,
                                int32* uniform)
OPEN3D_EXPORT_VECTORIZED_LAMBDA(LambdaKernel9,
                                Lambda9,
                                int64* uniform,
                                bool* uniform,
                                uint8* uniform,
                                int8* uniform,
                                uint16* uniform,
                                int16* uniform,
                                uint32* uniform,
                                int32* uniform,
                                uint64* uniform)

#define GET_INITIAL_VALUE(T, value) \
    int64 OPEN3D_SPECIALIZED(T, GetInitialValue)() { return value; }

GET_INITIAL_VALUE(bool, -1)
GET_INITIAL_VALUE(uint8_t, -2)
GET_INITIAL_VALUE(int8_t, -3)
GET_INITIAL_VALUE(uint16_t, -4)
GET_INITIAL_VALUE(int16_t, -5)
GET_INITIAL_VALUE(uint32_t, -6)
GET_INITIAL_VALUE(int32_t, -7)
GET_INITIAL_VALUE(uint64_t, -8)
GET_INITIAL_VALUE(int64_t, -9)
GET_INITIAL_VALUE(float, -10)
GET_INITIAL_VALUE(double, -11)

#undef GET_INITIAL_VALUE

#define TEMPLATE(T)                                                         \
    static inline void OPEN3D_SPECIALIZED(T, TemplateLambda1)(              \
            int64 idx, int64 * uniform v) {                                 \
        v[idx] = idx == 0 ? OPEN3D_SPECIALIZED(T, GetInitialValue)() : idx; \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                             \
    static inline void OPEN3D_SPECIALIZED(T, TemplateLambda2)(  \
            int64 idx, int64 * uniform v, bool* uniform arg2) { \
        OPEN3D_SPECIALIZED(T, TemplateLambda1)(idx, v);         \
        *arg2 = true;                                           \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                            \
    static inline void OPEN3D_SPECIALIZED(T, TemplateLambda3)( \
            int64 idx, int64 * uniform v, bool* uniform arg2,  \
            uint8* uniform arg3) {                             \
        OPEN3D_SPECIALIZED(T, TemplateLambda2)(idx, v, arg2);  \
        *arg3 = 3;                                             \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                 \
    static inline void OPEN3D_SPECIALIZED(T, TemplateLambda4)(      \
            int64 idx, int64 * uniform v, bool* uniform arg2,       \
            uint8* uniform arg3, int8* uniform arg4) {              \
        OPEN3D_SPECIALIZED(T, TemplateLambda3)(idx, v, arg2, arg3); \
        *arg4 = 4;                                                  \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                          \
    static inline void OPEN3D_SPECIALIZED(T, TemplateLambda5)(               \
            int64 idx, int64 * uniform v, bool* uniform arg2,                \
            uint8* uniform arg3, int8* uniform arg4, uint16* uniform arg5) { \
        OPEN3D_SPECIALIZED(T, TemplateLambda4)(idx, v, arg2, arg3, arg4);    \
        *arg5 = 5;                                                           \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                        \
    static inline void OPEN3D_SPECIALIZED(T, TemplateLambda6)(             \
            int64 idx, int64 * uniform v, bool* uniform arg2,              \
            uint8* uniform arg3, int8* uniform arg4, uint16* uniform arg5, \
            int16* uniform arg6) {                                         \
        OPEN3D_SPECIALIZED(T, TemplateLambda5)                             \
        (idx, v, arg2, arg3, arg4, arg5);                                  \
        *arg6 = 6;                                                         \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                        \
    static inline void OPEN3D_SPECIALIZED(T, TemplateLambda7)(             \
            int64 idx, int64 * uniform v, bool* uniform arg2,              \
            uint8* uniform arg3, int8* uniform arg4, uint16* uniform arg5, \
            int16* uniform arg6, uint32* uniform arg7) {                   \
        OPEN3D_SPECIALIZED(T, TemplateLambda6)                             \
        (idx, v, arg2, arg3, arg4, arg5, arg6);                            \
        *arg7 = 7;                                                         \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                           \
    static inline void OPEN3D_SPECIALIZED(T, TemplateLambda8)(                \
            int64 idx, int64 * uniform v, bool* uniform arg2,                 \
            uint8* uniform arg3, int8* uniform arg4, uint16* uniform arg5,    \
            int16* uniform arg6, uint32* uniform arg7, int32* uniform arg8) { \
        OPEN3D_SPECIALIZED(T, TemplateLambda7)                                \
        (idx, v, arg2, arg3, arg4, arg5, arg6, arg7);                         \
        *arg8 = 8;                                                            \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                         \
    static inline void OPEN3D_SPECIALIZED(T, TemplateLambda9)(              \
            int64 idx, int64 * uniform v, bool* uniform arg2,               \
            uint8* uniform arg3, int8* uniform arg4, uint16* uniform arg5,  \
            int16* uniform arg6, uint32* uniform arg7, int32* uniform arg8, \
            uint64* uniform arg9) {                                         \
        OPEN3D_SPECIALIZED(T, TemplateLambda8)                              \
        (idx, v, arg2, arg3, arg4, arg5, arg6, arg7, arg8);                 \
        *arg9 = 9;                                                          \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA(TemplateLambdaKernel1,
                                         TemplateLambda1,
                                         int64* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA(TemplateLambdaKernel2,
                                         TemplateLambda2,
                                         int64* uniform,
                                         bool* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA(TemplateLambdaKernel3,
                                         TemplateLambda3,
                                         int64* uniform,
                                         bool* uniform,
                                         uint8* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA(TemplateLambdaKernel4,
                                         TemplateLambda4,
                                         int64* uniform,
                                         bool* uniform,
                                         uint8* uniform,
                                         int8* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA(TemplateLambdaKernel5,
                                         TemplateLambda5,
                                         int64* uniform,
                                         bool* uniform,
                                         uint8* uniform,
                                         int8* uniform,
                                         uint16* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA(TemplateLambdaKernel6,
                                         TemplateLambda6,
                                         int64* uniform,
                                         bool* uniform,
                                         uint8* uniform,
                                         int8* uniform,
                                         uint16* uniform,
                                         int16* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA(TemplateLambdaKernel7,
                                         TemplateLambda7,
                                         int64* uniform,
                                         bool* uniform,
                                         uint8* uniform,
                                         int8* uniform,
                                         uint16* uniform,
                                         int16* uniform,
                                         uint32* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA(TemplateLambdaKernel8,
                                         TemplateLambda8,
                                         int64* uniform,
                                         bool* uniform,
                                         uint8* uniform,
                                         int8* uniform,
                                         uint16* uniform,
                                         int16* uniform,
                                         uint32* uniform,
                                         int32* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA(TemplateLambdaKernel9,
                                         TemplateLambda9,
                                         int64* uniform,
                                         bool* uniform,
                                         uint8* uniform,
                                         int8* uniform,
                                         uint16* uniform,
                                         int16* uniform,
                                         uint32* uniform,
                                         int32* uniform,
                                         uint64* uniform)
