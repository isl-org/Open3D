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
