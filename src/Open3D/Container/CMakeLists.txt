set (KERNEL_SRC
    Kernel/UnaryEW.cpp
    Kernel/UnaryEWCPU.cpp
)

set (KERNEL_CUDA_SRC
    Kernel/UnaryEWCUDA.cu
)

set (CONTAINER_SRC
    AdvancedIndexing.cpp
    Broadcast.cpp
    MemoryManager.cpp
    MemoryManagerCPU.cpp
    MemoryManagerCUDA.cu
    Tensor.cpp
)

set (CONTAINER_CUDA_SRC
    MemoryManagerCUDA.cu
)

if (BUILD_CUDA_MODULE)
    set (ALL_CONTAINER_SRC
        ${CONTAINER_SRC}
        ${CONTAINER_CUDA_SRC}
        ${KERNEL_SRC}
        ${KERNEL_CUDA_SRC}
    )
else()
    set (ALL_CONTAINER_SRC
        ${CONTAINER_SRC}
        ${KERNEL_SRC}
    )
endif()

# Create object library
add_library(Container OBJECT ${ALL_CONTAINER_SRC})
if (BUILD_CUDA_MODULE)
    target_include_directories(Container PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
