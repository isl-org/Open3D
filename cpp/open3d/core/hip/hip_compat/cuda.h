// CUDA-toolkit include shim for the HIP build. On the NVIDIA build this dir is
// not on the include path, so the real <cuda.h> is used unchanged.
#pragma once
#include "open3d/core/hip/CUDAToHIP.h"
