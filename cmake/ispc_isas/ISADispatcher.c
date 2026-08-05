// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <stdio.h>

#include "ISADispatcher_ispc.h"

const char* ToTargetString(enum ISATarget target) {
    switch (target) {
        /* x86 */
        case SSE2:
            return "sse2";
        case SSE4:
            return "sse4";
        case AVX:
            return "avx1";
        case AVX2:
            return "avx2";
        case AVX512KNL:
            return "avx512knl";
        case AVX512SKX:
            return "avx512skx";

        /* Return invalid string for unknown ISAs */
        default:
            return "UNKNOWN";
    }
}

int main() {
    printf("%s-i%dx%d", ToTargetString(GetISATarget()), GetISAElementWidth(),
           GetISAWidth());
    return 0;
}
