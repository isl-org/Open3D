message(STATUS "Building Tensorflow ops")

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if(BUILD_CUDA_MODULE)
    set(CUDA_ENABLED YES)
    message(STATUS "Building Tensorflow ops with CUDA")
endif()

find_package(Tensorflow REQUIRED)
find_package(TBB REQUIRED)
find_package(nanoflann REQUIRED)
find_package(parallelstl REQUIRED)
if(CUDA_ENABLED)
    find_package(CUB REQUIRED)
    find_package(CUTLASS REQUIRED)
endif()

if(CUDA_ENABLED)
    file(
        GLOB_RECURSE
        open3d_tf_ops_SOURCES
        "*.cpp"
        "*.cu"
        "../*/Detail/*.cc"
        "../*/Detail/*.cpp"
        "../*/Detail/*.cu")
else()
    file(GLOB_RECURSE open3d_tf_ops_SOURCES "*.cpp" "../*/Detail/*.cc"
         "../*/Detail/*.cpp")
endif()

add_library(open3d_tf_ops SHARED ${open3d_tf_ops_SOURCES})
open3d_set_global_properties(open3d_tf_ops)
# open3d_show_and_abort_on_warning(open3d_tf_ops)

# Do not add "lib" prefix
set_target_properties(open3d_tf_ops PROPERTIES PREFIX "")
set_target_properties(open3d_tf_ops PROPERTIES DEBUG_POSTFIX "_debug")

if(CUDA_ENABLED)
    target_include_directories(open3d_tf_ops PRIVATE ${CUB_INCLUDE_DIR}
                                                     ${CUTLASS_INCLUDE_DIR})
endif()

# Silence warnings caused by tensorflow's Eigen Travis has a different version
# of protobuf in /usr/local/include TBB is in /usr/local/include, so it needs to
# be after Tensorflow
target_include_directories(open3d_tf_ops SYSTEM PRIVATE
    ${Tensorflow_INCLUDE_DIR}
    ${TBB_INCLUDE_DIR}
    ${nanoflann_INCLUDE_DIR}
    ${parallelstl_INCLUDE_DIR}
)

target_link_libraries(open3d_tf_ops PRIVATE
    ${CMAKE_PROJECT_NAME}
    ${TBB_LIBRARY}
    ${TBB_MALLOC_LIBRARY}
    ${Tensorflow_FRAMEWORK_LIB}
    ${FMT_TARGET}
)
