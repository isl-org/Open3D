# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE ALL_HEADER_FILES "*.h")

# create object library
if (CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCE_FILES "*.cu")
    add_library(Odometry OBJECT ${ALL_SOURCE_FILES}
                             ${ALL_HEADER_FILES}
                             ${CUDA_SOURCE_FILES})
    set_target_properties(Odometry PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    add_library(Odometry OBJECT ${ALL_SOURCE_FILES} ${ALL_HEADER_FILES})
endif()
