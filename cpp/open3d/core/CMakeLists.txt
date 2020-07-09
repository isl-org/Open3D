set (KERNEL_SRC
    op/kernel/IndexGetSet.cpp
    op/kernel/IndexGetSetCPU.cpp
    op/kernel/NonZero.cpp
    op/kernel/NonZeroCPU.cpp
    op/kernel/UnaryEW.cpp
    op/kernel/UnaryEWCPU.cpp
    op/kernel/BinaryEW.cpp
    op/kernel/BinaryEWCPU.cpp
    op/kernel/Reduction.cpp
    op/kernel/ReductionCPU.cpp
)

set (KERNEL_CUDA_SRC
    op/kernel/IndexGetSetCUDA.cu
    op/kernel/NonZeroCUDA.cu
    op/kernel/UnaryEWCUDA.cu
    op/kernel/BinaryEWCUDA.cu
    op/kernel/ReductionCUDA.cu
)

set (CORE_SRC
    AdvancedIndexing.cpp
    ShapeUtil.cpp
    CUDAUtils.cpp
    Indexer.cpp
    MemoryManager.cpp
    MemoryManagerCPU.cpp
    MemoryManagerCUDA.cu
    Tensor.cpp
    TensorKey.cpp
    TensorList.cpp
)

set (CORE_CUDA_SRC
    MemoryManagerCUDA.cu
)

if (BUILD_CUDA_MODULE)
    set (ALL_CORE_SRC
        ${CORE_SRC}
        ${CORE_CUDA_SRC}
        ${KERNEL_SRC}
        ${KERNEL_CUDA_SRC}
    )
else()
    set (ALL_CORE_SRC
        ${CORE_SRC}
        ${KERNEL_SRC}
    )
endif()

# Create object library
add_library(core OBJECT ${ALL_CORE_SRC})
open3d_set_global_properties(core)
open3d_link_3rdparty_libraries(core)
open3d_show_and_abort_on_warning(core)

if (BUILD_CUDA_MODULE)
    target_include_directories(core SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
