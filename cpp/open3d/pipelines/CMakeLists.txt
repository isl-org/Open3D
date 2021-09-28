open3d_ispc_add_library(pipelines OBJECT)

target_sources(pipelines PRIVATE
    color_map/ColorMapUtils.cpp
    color_map/ImageWarpingField.cpp
    color_map/NonRigidOptimizer.cpp
    color_map/RigidOptimizer.cpp
)

target_sources(pipelines PRIVATE
    integration/ScalableTSDFVolume.cpp
    integration/UniformTSDFVolume.cpp
)

target_sources(pipelines PRIVATE
    odometry/Odometry.cpp
    odometry/RGBDOdometryJacobian.cpp
)

target_sources(pipelines PRIVATE
    registration/ColoredICP.cpp
    registration/CorrespondenceChecker.cpp
    registration/FastGlobalRegistration.cpp
    registration/Feature.cpp
    registration/GeneralizedICP.cpp
    registration/GlobalOptimization.cpp
    registration/PoseGraph.cpp
    registration/Registration.cpp
    registration/RobustKernel.cpp
    registration/TransformationEstimation.cpp
)

open3d_show_and_abort_on_warning(pipelines)
open3d_set_global_properties(pipelines)
open3d_set_open3d_lib_properties(pipelines)
open3d_link_3rdparty_libraries(pipelines)
