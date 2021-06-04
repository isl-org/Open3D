add_library(tpipelines_kernel OBJECT)

target_sources(tpipelines_kernel  PRIVATE
    ComputeTransform.cpp
    ComputeTransformCPU.cpp
    FillInLinearSystem.cpp
    FillInLinearSystemCPU.cpp
    RGBDOdometry.cpp
    RGBDOdometryCPU.cpp
    TransformationConverter.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(tpipelines_kernel  PRIVATE
        ComputeTransformCUDA.cu
        FillInLinearSystemCUDA.cu
        RGBDOdometryCUDA.cu
        TransformationConverter.cu
    )
endif()

open3d_show_and_abort_on_warning(tpipelines_kernel)
open3d_set_global_properties(tpipelines_kernel)
open3d_set_open3d_lib_properties(tpipelines_kernel HIDDEN)
open3d_link_3rdparty_libraries(tpipelines_kernel)

if(BUILD_CUDA_MODULE)
    target_include_directories(tpipelines_kernel SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    find_package(CUB REQUIRED)
    target_include_directories(tpipelines_kernel SYSTEM PRIVATE ${CUB_INCLUDE_DIR})
endif()
