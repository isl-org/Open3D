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

#include "open3d/core/ParallelFor.isph"

static inline void ISPCFunction1(int64_t idx, int64_t* uniform v) {
    v[idx] = idx;
}
static inline void ISPCFunction2(int64_t idx,
                                 int64_t* uniform v,
                                 bool* uniform arg2) {
    ISPCFunction1(idx, v);
    *arg2 = true;
}
static inline void ISPCFunction3(int64_t idx,
                                 int64_t* uniform v,
                                 bool* uniform arg2,
                                 uint8_t* uniform arg3) {
    ISPCFunction2(idx, v, arg2);
    *arg3 = 3;
}
static inline void ISPCFunction4(int64_t idx,
                                 int64_t* uniform v,
                                 bool* uniform arg2,
                                 uint8_t* uniform arg3,
                                 int8_t* uniform arg4) {
    ISPCFunction3(idx, v, arg2, arg3);
    *arg4 = 4;
}
static inline void ISPCFunction5(int64_t idx,
                                 int64_t* uniform v,
                                 bool* uniform arg2,
                                 uint8_t* uniform arg3,
                                 int8_t* uniform arg4,
                                 uint16_t* uniform arg5) {
    ISPCFunction4(idx, v, arg2, arg3, arg4);
    *arg5 = 5;
}
static inline void ISPCFunction6(int64_t idx,
                                 int64_t* uniform v,
                                 bool* uniform arg2,
                                 uint8_t* uniform arg3,
                                 int8_t* uniform arg4,
                                 uint16_t* uniform arg5,
                                 int16_t* uniform arg6) {
    ISPCFunction5(idx, v, arg2, arg3, arg4, arg5);
    *arg6 = 6;
}
static inline void ISPCFunction7(int64_t idx,
                                 int64_t* uniform v,
                                 bool* uniform arg2,
                                 uint8_t* uniform arg3,
                                 int8_t* uniform arg4,
                                 uint16_t* uniform arg5,
                                 int16_t* uniform arg6,
                                 uint32_t* uniform arg7) {
    ISPCFunction6(idx, v, arg2, arg3, arg4, arg5, arg6);
    *arg7 = 7;
}
static inline void ISPCFunction8(int64_t idx,
                                 int64_t* uniform v,
                                 bool* uniform arg2,
                                 uint8_t* uniform arg3,
                                 int8_t* uniform arg4,
                                 uint16_t* uniform arg5,
                                 int16_t* uniform arg6,
                                 uint32_t* uniform arg7,
                                 int32_t* uniform arg8) {
    ISPCFunction7(idx, v, arg2, arg3, arg4, arg5, arg6, arg7);
    *arg8 = 8;
}

static inline void ISPCFunction9(int64_t idx,
                                 int64_t* uniform v,
                                 bool* uniform arg2,
                                 uint8_t* uniform arg3,
                                 int8_t* uniform arg4,
                                 uint16_t* uniform arg5,
                                 int16_t* uniform arg6,
                                 uint32_t* uniform arg7,
                                 int32_t* uniform arg8,
                                 uint64_t* uniform arg9) {
    ISPCFunction8(idx, v, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    *arg9 = 9;
}

OPEN3D_EXPORT_VECTORIZED(ISPCKernel1, ISPCFunction1, int64_t* uniform)
OPEN3D_EXPORT_VECTORIZED(ISPCKernel2,
                         ISPCFunction2,
                         int64_t* uniform,
                         bool* uniform)
OPEN3D_EXPORT_VECTORIZED(ISPCKernel3,
                         ISPCFunction3,
                         int64_t* uniform,
                         bool* uniform,
                         uint8_t* uniform)
OPEN3D_EXPORT_VECTORIZED(ISPCKernel4,
                         ISPCFunction4,
                         int64_t* uniform,
                         bool* uniform,
                         uint8_t* uniform,
                         int8_t* uniform)
OPEN3D_EXPORT_VECTORIZED(ISPCKernel5,
                         ISPCFunction5,
                         int64_t* uniform,
                         bool* uniform,
                         uint8_t* uniform,
                         int8_t* uniform,
                         uint16_t* uniform)
OPEN3D_EXPORT_VECTORIZED(ISPCKernel6,
                         ISPCFunction6,
                         int64_t* uniform,
                         bool* uniform,
                         uint8_t* uniform,
                         int8_t* uniform,
                         uint16_t* uniform,
                         int16_t* uniform)
OPEN3D_EXPORT_VECTORIZED(ISPCKernel7,
                         ISPCFunction7,
                         int64_t* uniform,
                         bool* uniform,
                         uint8_t* uniform,
                         int8_t* uniform,
                         uint16_t* uniform,
                         int16_t* uniform,
                         uint32_t* uniform)
OPEN3D_EXPORT_VECTORIZED(ISPCKernel8,
                         ISPCFunction8,
                         int64_t* uniform,
                         bool* uniform,
                         uint8_t* uniform,
                         int8_t* uniform,
                         uint16_t* uniform,
                         int16_t* uniform,
                         uint32_t* uniform,
                         int32_t* uniform)
OPEN3D_EXPORT_VECTORIZED(ISPCKernel9,
                         ISPCFunction9,
                         int64_t* uniform,
                         bool* uniform,
                         uint8_t* uniform,
                         int8_t* uniform,
                         uint16_t* uniform,
                         int16_t* uniform,
                         uint32_t* uniform,
                         int32_t* uniform,
                         uint64_t* uniform)

#define GET_INITIAL_VALUE(T, value) \
    int64_t OPEN3D_SPECIALIZED(T, GetInitialValue)() { return value; }

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
    static inline void OPEN3D_SPECIALIZED(T, TemplateISPCFunction1)(        \
            int64_t idx, int64_t * uniform v) {                             \
        v[idx] = idx == 0 ? OPEN3D_SPECIALIZED(T, GetInitialValue)() : idx; \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                  \
    static inline void OPEN3D_SPECIALIZED(T, TemplateISPCFunction2)( \
            int64_t idx, int64_t * uniform v, bool* uniform arg2) {  \
        OPEN3D_SPECIALIZED(T, TemplateISPCFunction1)(idx, v);        \
        *arg2 = true;                                                \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                  \
    static inline void OPEN3D_SPECIALIZED(T, TemplateISPCFunction3)( \
            int64_t idx, int64_t * uniform v, bool* uniform arg2,    \
            uint8_t* uniform arg3) {                                 \
        OPEN3D_SPECIALIZED(T, TemplateISPCFunction2)(idx, v, arg2);  \
        *arg3 = 3;                                                   \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                       \
    static inline void OPEN3D_SPECIALIZED(T, TemplateISPCFunction4)(      \
            int64_t idx, int64_t * uniform v, bool* uniform arg2,         \
            uint8_t* uniform arg3, int8_t* uniform arg4) {                \
        OPEN3D_SPECIALIZED(T, TemplateISPCFunction3)(idx, v, arg2, arg3); \
        *arg4 = 4;                                                        \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                  \
    static inline void OPEN3D_SPECIALIZED(T, TemplateISPCFunction5)( \
            int64_t idx, int64_t * uniform v, bool* uniform arg2,    \
            uint8_t* uniform arg3, int8_t* uniform arg4,             \
            uint16_t* uniform arg5) {                                \
        OPEN3D_SPECIALIZED(T, TemplateISPCFunction4)                 \
        (idx, v, arg2, arg3, arg4);                                  \
        *arg5 = 5;                                                   \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                  \
    static inline void OPEN3D_SPECIALIZED(T, TemplateISPCFunction6)( \
            int64_t idx, int64_t * uniform v, bool* uniform arg2,    \
            uint8_t* uniform arg3, int8_t* uniform arg4,             \
            uint16_t* uniform arg5, int16_t* uniform arg6) {         \
        OPEN3D_SPECIALIZED(T, TemplateISPCFunction5)                 \
        (idx, v, arg2, arg3, arg4, arg5);                            \
        *arg6 = 6;                                                   \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                  \
    static inline void OPEN3D_SPECIALIZED(T, TemplateISPCFunction7)( \
            int64_t idx, int64_t * uniform v, bool* uniform arg2,    \
            uint8_t* uniform arg3, int8_t* uniform arg4,             \
            uint16_t* uniform arg5, int16_t* uniform arg6,           \
            uint32_t* uniform arg7) {                                \
        OPEN3D_SPECIALIZED(T, TemplateISPCFunction6)                 \
        (idx, v, arg2, arg3, arg4, arg5, arg6);                      \
        *arg7 = 7;                                                   \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                  \
    static inline void OPEN3D_SPECIALIZED(T, TemplateISPCFunction8)( \
            int64_t idx, int64_t * uniform v, bool* uniform arg2,    \
            uint8_t* uniform arg3, int8_t* uniform arg4,             \
            uint16_t* uniform arg5, int16_t* uniform arg6,           \
            uint32_t* uniform arg7, int32_t* uniform arg8) {         \
        OPEN3D_SPECIALIZED(T, TemplateISPCFunction7)                 \
        (idx, v, arg2, arg3, arg4, arg5, arg6, arg7);                \
        *arg8 = 8;                                                   \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

#define TEMPLATE(T)                                                  \
    static inline void OPEN3D_SPECIALIZED(T, TemplateISPCFunction9)( \
            int64_t idx, int64_t * uniform v, bool* uniform arg2,    \
            uint8_t* uniform arg3, int8_t* uniform arg4,             \
            uint16_t* uniform arg5, int16_t* uniform arg6,           \
            uint32_t* uniform arg7, int32_t* uniform arg8,           \
            uint64_t* uniform arg9) {                                \
        OPEN3D_SPECIALIZED(T, TemplateISPCFunction8)                 \
        (idx, v, arg2, arg3, arg4, arg5, arg6, arg7, arg8);          \
        *arg9 = 9;                                                   \
    }
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

OPEN3D_EXPORT_TEMPLATE_VECTORIZED(TemplateISPCKernel1,
                                  TemplateISPCFunction1,
                                  int64_t* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED(TemplateISPCKernel2,
                                  TemplateISPCFunction2,
                                  int64_t* uniform,
                                  bool* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED(TemplateISPCKernel3,
                                  TemplateISPCFunction3,
                                  int64_t* uniform,
                                  bool* uniform,
                                  uint8_t* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED(TemplateISPCKernel4,
                                  TemplateISPCFunction4,
                                  int64_t* uniform,
                                  bool* uniform,
                                  uint8_t* uniform,
                                  int8_t* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED(TemplateISPCKernel5,
                                  TemplateISPCFunction5,
                                  int64_t* uniform,
                                  bool* uniform,
                                  uint8_t* uniform,
                                  int8_t* uniform,
                                  uint16_t* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED(TemplateISPCKernel6,
                                  TemplateISPCFunction6,
                                  int64_t* uniform,
                                  bool* uniform,
                                  uint8_t* uniform,
                                  int8_t* uniform,
                                  uint16_t* uniform,
                                  int16_t* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED(TemplateISPCKernel7,
                                  TemplateISPCFunction7,
                                  int64_t* uniform,
                                  bool* uniform,
                                  uint8_t* uniform,
                                  int8_t* uniform,
                                  uint16_t* uniform,
                                  int16_t* uniform,
                                  uint32_t* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED(TemplateISPCKernel8,
                                  TemplateISPCFunction8,
                                  int64_t* uniform,
                                  bool* uniform,
                                  uint8_t* uniform,
                                  int8_t* uniform,
                                  uint16_t* uniform,
                                  int16_t* uniform,
                                  uint32_t* uniform,
                                  int32_t* uniform)
OPEN3D_EXPORT_TEMPLATE_VECTORIZED(TemplateISPCKernel9,
                                  TemplateISPCFunction9,
                                  int64_t* uniform,
                                  bool* uniform,
                                  uint8_t* uniform,
                                  int8_t* uniform,
                                  uint16_t* uniform,
                                  int16_t* uniform,
                                  uint32_t* uniform,
                                  int32_t* uniform,
                                  uint64_t* uniform)
