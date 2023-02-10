// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/ISAInfo.h"

#ifdef BUILD_ISPC_MODULE
#include "ISAInfo_ispc.h"
#endif

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

struct ISAInfo::Impl {
    ISATarget target_;
};

static ISATarget GetSelectedISATarget() {
#ifdef BUILD_ISPC_MODULE
    ispc::ISAInfo info;
    ispc::open3d_utility_GetISAInfo(&info);
    return static_cast<ISATarget>(info.target);
#else
    return ISATarget::DISABLED;
#endif
}

static std::string ToString(ISATarget target) {
    switch (target) {
        /* x86 */
        case ISATarget::SSE2:
            return "SSE2";
        case ISATarget::SSE4:
            return "SSE4";
        case ISATarget::AVX:
            return "AVX";
        case ISATarget::AVX2:
            return "AVX2";
        case ISATarget::AVX512KNL:
            return "AVX512KNL";
        case ISATarget::AVX512SKX:
            return "AVX512SKX";
        /* ARM */
        case ISATarget::NEON:
            return "NEON";
        /* GPU */
        case ISATarget::GENX:
            return "GENX";
        /* Special values */
        case ISATarget::UNKNOWN:
            return "UNKNOWN";

        /* Additional value for disabled support */
        case ISATarget::DISABLED:
            return "DISABLED";

        default:
            utility::LogError("Unknown target {}", static_cast<int>(target));
    }
}

ISAInfo::ISAInfo() : impl_(new ISAInfo::Impl()) {
    impl_->target_ = GetSelectedISATarget();
}

ISAInfo& ISAInfo::GetInstance() {
    static ISAInfo instance;
    return instance;
}

ISATarget ISAInfo::SelectedTarget() const { return impl_->target_; }

void ISAInfo::Print() const {
    utility::LogInfo("ISAInfo: {} instruction set used in ISPC code.",
                     ToString(SelectedTarget()));
}

}  // namespace utility
}  // namespace open3d
