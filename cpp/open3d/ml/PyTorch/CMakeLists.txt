message(STATUS "Building Pytorch ops")

if(BUILD_CUDA_MODULE)
    message(STATUS "Building Pytorch ops with CUDA")
endif()

find_package(Pytorch REQUIRED)
find_package(TBB REQUIRED)
find_package(nanoflann REQUIRED)
find_package(parallelstl REQUIRED)
if(BUILD_CUDA_MODULE)
    find_package(CUB REQUIRED)
    # find_package(CUTLASS REQUIRED)
endif()

if(BUILD_CUDA_MODULE)
    file(
        GLOB_RECURSE
        open3d_torch_ops_SOURCES
        "*.cpp"
        "*.cu"
        "../impl/*/*.cc"
        "../impl/*/*.cpp"
        "../impl/*/*.cu")
else()
    file(GLOB_RECURSE open3d_torch_ops_SOURCES "*.cpp" "../impl/*/*.cc"
         "../impl/*/*.cpp")
endif()
message(STATUS ${open3d_torch_ops_SOURCES})

add_library(open3d_torch_ops SHARED ${open3d_torch_ops_SOURCES})
open3d_set_global_properties(open3d_torch_ops)
open3d_show_and_abort_on_warning(open3d_torch_ops)

# Do not add "lib" prefix
set_target_properties(open3d_torch_ops PROPERTIES PREFIX "")
set_target_properties(open3d_torch_ops PROPERTIES DEBUG_POSTFIX "_debug")

set( open3d_torch_ops_DEFINITIONS )
if(BUILD_CUDA_MODULE)
    target_include_directories(open3d_torch_ops SYSTEM PRIVATE
        ${CUB_INCLUDE_DIR}
        # ${CUTLASS_INCLUDE_DIR}
    )
    list(APPEND open3d_torch_ops_DEFINITIONS "BUILD_CUDA_MODULE")
endif()
target_compile_definitions(open3d_torch_ops PRIVATE "${open3d_torch_ops_DEFINITIONS}")

target_include_directories(open3d_torch_ops SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/cpp
    ${TBB_INCLUDE_DIR}
    ${nanoflann_INCLUDE_DIR}
    ${parallelstl_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(open3d_torch_ops PRIVATE
    ${TBB_LIBRARY}
    ${TBB_MALLOC_LIBRARY}
    ${TORCH_LIBRARIES}
    ${EIGEN3_TARGET}
    ${FMT_TARGET}
)

