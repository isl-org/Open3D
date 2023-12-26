// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/kernel/BinaryEW.h"

#include <vector>

#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

const std::unordered_set<BinaryEWOpCode, utility::hash_enum_class>
        s_boolean_binary_ew_op_codes{
                BinaryEWOpCode::LogicalAnd, BinaryEWOpCode::LogicalOr,
                BinaryEWOpCode::LogicalXor, BinaryEWOpCode::Gt,
                BinaryEWOpCode::Lt,         BinaryEWOpCode::Ge,
                BinaryEWOpCode::Le,         BinaryEWOpCode::Eq,
                BinaryEWOpCode::Ne,
        };

void BinaryEW(const Tensor& lhs,
              const Tensor& rhs,
              Tensor& dst,
              BinaryEWOpCode op_code) {
    // lhs, rhs and dst must be on the same device.
    for (auto device :
         std::vector<Device>({rhs.GetDevice(), dst.GetDevice()})) {
        if (lhs.GetDevice() != device) {
            utility::LogError("Device mismatch {} != {}.",
                              lhs.GetDevice().ToString(), device.ToString());
        }
    }

    // broadcast(lhs.shape, rhs.shape) must be dst.shape.
    const SizeVector broadcasted_input_shape =
            shape_util::BroadcastedShape(lhs.GetShape(), rhs.GetShape());
    if (broadcasted_input_shape != dst.GetShape()) {
        utility::LogError(
                "The broadcasted input shape {} does not match the output "
                "shape {}.",
                broadcasted_input_shape, dst.GetShape());
    }

    if (lhs.IsCPU()) {
        BinaryEWCPU(lhs, rhs, dst, op_code);
    } else if (lhs.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        BinaryEWCUDA(lhs, rhs, dst, op_code);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("BinaryEW: Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
