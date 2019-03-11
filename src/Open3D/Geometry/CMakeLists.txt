# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
if (CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCE_FILES "*.cu")
    add_library(Geometry OBJECT ${ALL_SOURCE_FILES} ${CUDA_SOURCE_FILES})
    set_target_properties(Geometry PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    add_library(Geometry OBJECT ${ALL_SOURCE_FILES})
endif()
