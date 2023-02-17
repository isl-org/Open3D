add_subdirectory(kernel)

open3d_ispc_add_library(tpipelines OBJECT)

target_sources(tpipelines PRIVATE
    odometry/RGBDOdometry.cpp
)

target_sources(tpipelines PRIVATE
    registration/Registration.cpp
    registration/TransformationEstimation.cpp
    registration/Feature.cpp
)

target_sources(tpipelines PRIVATE
    slac/ControlGrid.cpp
    slac/SLACOptimizer.cpp
    slac/Visualization.cpp
)

target_sources(tpipelines PRIVATE
    slam/Model.cpp
)

open3d_show_and_abort_on_warning(tpipelines)
open3d_set_global_properties(tpipelines)
open3d_set_open3d_lib_properties(tpipelines)
open3d_link_3rdparty_libraries(tpipelines)

if(BUILD_CUDA_MODULE)
    target_include_directories(tpipelines SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
