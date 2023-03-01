// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Helper.isph"

struct NotAutomaticallyExportedStruct {
    int32_t i;
    uint8_t b;
};

OPEN3D_ENSURE_EXPORTED(NotAutomaticallyExportedStruct)

export void GetCharBit(uniform int32_t* uniform value) { *value = CHAR_BIT; }
