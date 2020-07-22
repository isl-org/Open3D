set (KERNEL_SRC
    kernel/IndexGetSet.cpp
    kernel/IndexGetSetCPU.cpp
    kernel/NonZero.cpp
    kernel/NonZeroCPU.cpp
    kernel/UnaryEW.cpp
    kernel/UnaryEWCPU.cpp
    kernel/BinaryEW.cpp
    kernel/BinaryEWCPU.cpp
    kernel/Reduction.cpp
    kernel/ReductionCPU.cpp
)

set (KERNEL_CUDA_SRC
    kernel/IndexGetSetCUDA.cu
    kernel/NonZeroCUDA.cu
    kernel/UnaryEWCUDA.cu
    kernel/BinaryEWCUDA.cu
    kernel/ReductionCUDA.cu
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
