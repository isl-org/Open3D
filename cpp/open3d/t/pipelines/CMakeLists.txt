add_library(tpipelines OBJECT)

target_sources(tpipelines PRIVATE
    kernel/ComputeTransform.cpp
    kernel/ComputeTransformCPU.cpp
    kernel/FillInLinearSystem.cpp
    kernel/FillInLinearSystemCPU.cpp
    kernel/RGBDOdometry.cpp
    kernel/RGBDOdometryCPU.cpp
    kernel/TransformationConverter.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(tpipelines PRIVATE
        kernel/ComputeTransformCUDA.cu
        kernel/FillInLinearSystemCUDA.cu
        kernel/RGBDOdometryCUDA.cu
        kernel/TransformationConverter.cu
    )
endif()

target_sources(tpipelines PRIVATE
    odometry/RGBDOdometry.cpp
)

target_sources(tpipelines PRIVATE
    registration/Registration.cpp
    registration/TransformationEstimation.cpp
)

target_sources(tpipelines PRIVATE
    slac/ControlGrid.cpp
    slac/SLACOptimizer.cpp
    slac/Visualization.cpp
)

target_sources(tpipelines PRIVATE
    voxelhashing/Model.cpp
)

open3d_show_and_abort_on_warning(tpipelines)
open3d_set_global_properties(tpipelines)
open3d_set_open3d_lib_properties(tpipelines)
open3d_link_3rdparty_libraries(tpipelines)

if(BUILD_CUDA_MODULE)
    target_include_directories(tpipelines SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    find_package(CUB REQUIRED)
    target_include_directories(tpipelines SYSTEM PRIVATE ${CUB_INCLUDE_DIR})
endif()
