
#pragma once

#include "cassert"

#ifdef __CUDACC__
#define OPEN3D_FUNC_DECL __host__ __device__
#else
#define OPEN3D_FUNC_DECL
#endif
