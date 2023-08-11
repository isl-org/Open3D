// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include <memory>

namespace open3d {
namespace utility {

/// Set of known ISA targets.
enum class ISATarget {
    /* x86 */
    SSE2 = 100,
    SSE4 = 101,
    AVX = 102,
    AVX2 = 103,
    AVX512KNL = 104,
    AVX512SKX = 105,
    /* ARM */
    NEON = 200,
    /* GPU */
    GENX = 300,
    /* Special values */
    UNKNOWN = -1,
    /* Additional value for disabled support */
    DISABLED = -100
};

/// \brief ISA information.
///
/// This provides information about kernel code written in ISPC.
class ISAInfo {
public:
    static ISAInfo& GetInstance();

    ~ISAInfo() = default;
    ISAInfo(const ISAInfo&) = delete;
    void operator=(const ISAInfo&) = delete;

    /// Returns the dispatched ISA target that will be used in kernel code.
    ISATarget SelectedTarget() const;

    /// Prints ISAInfo in the console.
    void Print() const;

private:
    ISAInfo();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace utility
}  // namespace open3d
