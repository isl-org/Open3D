#include "open3d/core/nns/DeviceDefs.cuh"
#include "open3d/core/nns/blockselect/BlockSelectImpl.cuh"

namespace open3d {
namespace core {

// warp Q to thread Q:
// 1, 1
// 32, 2
// 64, 3
// 128, 3
// 256, 4
// 512, 8
// 1024, 8
// 2048, 8

BLOCK_SELECT_DECL(float, true, 1);
BLOCK_SELECT_DECL(float, true, 32);
BLOCK_SELECT_DECL(float, true, 64);
BLOCK_SELECT_DECL(float, true, 128);
BLOCK_SELECT_DECL(float, true, 256);
BLOCK_SELECT_DECL(float, true, 512);
BLOCK_SELECT_DECL(float, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_DECL(float, true, 2048);
#endif

BLOCK_SELECT_DECL(float, false, 1);
BLOCK_SELECT_DECL(float, false, 32);
BLOCK_SELECT_DECL(float, false, 64);
BLOCK_SELECT_DECL(float, false, 128);
BLOCK_SELECT_DECL(float, false, 256);
BLOCK_SELECT_DECL(float, false, 512);
BLOCK_SELECT_DECL(float, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_DECL(float, false, 2048);
#endif

void runBlockSelectPair(float* inK,
                        int32_t* inV,
                        float* outK,
                        int32_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points,
                        cudaStream_t stream) {
    OPEN3D_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (dir) {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(float, true, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(float, true, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(float, true, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(float, true, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(float, true, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(float, true, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(float, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(float, true, 2048);
#endif
        }
    } else {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(float, false, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(float, false, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(float, false, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(float, false, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(float, false, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(float, false, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(float, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(float, false, 2048);
#endif
        }
    }
}

BLOCK_SELECT_DECL(double, true, 1);
BLOCK_SELECT_DECL(double, true, 32);
BLOCK_SELECT_DECL(double, true, 64);
BLOCK_SELECT_DECL(double, true, 128);
BLOCK_SELECT_DECL(double, true, 256);
BLOCK_SELECT_DECL(double, true, 512);
BLOCK_SELECT_DECL(double, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_DECL(double, true, 2048);
#endif

BLOCK_SELECT_DECL(double, false, 1);
BLOCK_SELECT_DECL(double, false, 32);
BLOCK_SELECT_DECL(double, false, 64);
BLOCK_SELECT_DECL(double, false, 128);
BLOCK_SELECT_DECL(double, false, 256);
BLOCK_SELECT_DECL(double, false, 512);
BLOCK_SELECT_DECL(double, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_DECL(double, false, 2048);
#endif
void runBlockSelectPair(double* inK,
                        int32_t* inV,
                        double* outK,
                        int32_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points,
                        cudaStream_t stream) {
    OPEN3D_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (dir) {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(double, true, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(double, true, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(double, true, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(double, true, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(double, true, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(double, true, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(double, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(double, true, 2048);
#endif
        }
    } else {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(double, false, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(double, false, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(double, false, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(double, false, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(double, false, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(double, false, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(double, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(double, false, 2048);
#endif
        }
    }
}

}  // namespace core
}  // namespace open3d