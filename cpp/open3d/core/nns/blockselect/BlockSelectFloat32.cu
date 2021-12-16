
#include "open3d/core/nns/blockselect/BlockSelectImpl.cuh"

namespace open3d {
namespace core {
BLOCK_SELECT_IMPL(float, true, 1, 1);
BLOCK_SELECT_IMPL(float, false, 1, 1);

BLOCK_SELECT_IMPL(float, true, 32, 2);
BLOCK_SELECT_IMPL(float, false, 32, 2);

BLOCK_SELECT_IMPL(float, true, 64, 3);
BLOCK_SELECT_IMPL(float, false, 64, 3);

BLOCK_SELECT_IMPL(float, true, 128, 3);
BLOCK_SELECT_IMPL(float, false, 128, 3);

BLOCK_SELECT_IMPL(float, true, 256, 4);
BLOCK_SELECT_IMPL(float, false, 256, 4);

BLOCK_SELECT_IMPL(float, true, 512, 8);
BLOCK_SELECT_IMPL(float, false, 512, 8);

BLOCK_SELECT_IMPL(float, true, 1024, 8);
BLOCK_SELECT_IMPL(float, false, 1024, 8);

#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_IMPL(float, true, 2048, 8);
BLOCK_SELECT_IMPL(float, false, 2048, 8);
#endif

}  // namespace core
}  // namespace open3d