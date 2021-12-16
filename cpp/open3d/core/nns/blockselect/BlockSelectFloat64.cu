
#include "open3d/core/nns/blockselect/BlockSelectImpl.cuh"

namespace open3d {
namespace core {

BLOCK_SELECT_IMPL(double, true, 1, 1);
BLOCK_SELECT_IMPL(double, false, 1, 1);

BLOCK_SELECT_IMPL(double, true, 32, 2);
BLOCK_SELECT_IMPL(double, false, 32, 2);

BLOCK_SELECT_IMPL(double, true, 64, 3);
BLOCK_SELECT_IMPL(double, false, 64, 3);

BLOCK_SELECT_IMPL(double, true, 128, 3);
BLOCK_SELECT_IMPL(double, false, 128, 3);

BLOCK_SELECT_IMPL(double, true, 256, 4);
BLOCK_SELECT_IMPL(double, false, 256, 4);

BLOCK_SELECT_IMPL(double, true, 512, 8);
BLOCK_SELECT_IMPL(double, false, 512, 8);

BLOCK_SELECT_IMPL(double, true, 1024, 8);
BLOCK_SELECT_IMPL(double, false, 1024, 8);

#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_IMPL(double, true, 2048, 8);
BLOCK_SELECT_IMPL(double, false, 2048, 8);
#endif

}  // namespace core
}  // namespace open3d