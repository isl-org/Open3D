add_subdirectory(cpp)

if (BUILD_CUDA_MODULE)
    # TODO: remove
    add_subdirectory(hashmap)
endif()
