# Build

# Common source code shared by CUDA and host
set (SRC_CPU
    MemoryManager.cpp
    MemoryManagerCPU.cpp
)

# Additional source code for when CUDA is enabled
set (SRC_GPU
    MemoryManagerGPU.cpp
)

# Create object library
if (BUILD_CUDA_MODULE)
    add_definitions(-DBUILD_CUDA_MODULE)
    cuda_add_library(Container ${SRC_CPU} ${SRC_GPU})
    target_include_directories(Container PUBLIC ${CUDA_INCLUDE_DIRS})
else()
    add_library(Container ${SRC_CPU})
endif()
