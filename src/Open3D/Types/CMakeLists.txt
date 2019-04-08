# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE ALL_HEADER_FILES "*.h")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# create object library
if (CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCE_FILES "*.cu")
    add_library(Types OBJECT ${ALL_SOURCE_FILES}
                             ${ALL_HEADER_FILES}
                             ${CUDA_SOURCE_FILES})
    set_target_properties(Types PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 11)
else()
    add_library(Types OBJECT ${ALL_SOURCE_FILES} ${ALL_HEADER_FILES})
endif()
