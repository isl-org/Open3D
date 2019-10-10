# Build
set (CONTAINER_SRC
    MemoryManager.cpp
    MemoryManagerCPU.cpp
    MemoryManagerCUDA.cu
    Tensor.cpp
)

if (BUILD_CUDA_MODULE)
    set (CONTAINER_SRC
        ${CONTAINER_SRC}
        MemoryManagerCUDA.cu
    )
endif()

# Create object library
add_library(Container OBJECT ${CONTAINER_SRC})
if (BUILD_CUDA_MODULE)
    target_include_directories(Container PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
