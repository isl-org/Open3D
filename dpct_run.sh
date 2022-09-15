#/bin/env bash

open3d_root=/home/yixing/repo/Open3D

dpct \
    --in-root ${open3d_root}/cpp \
    --out-root ${open3d_root}/build/out \
    --extra-arg="-DBUILD_CUDA_MODULE" \
    --extra-arg="-I${open3d_root}/cpp" \
    --extra-arg="-I${open3d_root}/build/fmt/include" \
    ${open3d_root}/cpp/open3d/core/kernel/ReductionCUDA.cu
