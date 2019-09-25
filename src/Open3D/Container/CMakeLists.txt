# Build
set (CONTAINER_SRC
    MemoryManager.cpp
)

# Create object library
add_library(Container OBJECT ${CONTAINER_SRC})
if (BUILD_CUDA_MODULE)
    add_definitions(-DBUILD_CUDA_MODULE)
    target_include_directories(Container PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
