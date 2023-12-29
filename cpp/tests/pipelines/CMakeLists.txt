target_sources(tests PRIVATE
    integration/ScalableTSDFVolume.cpp
    integration/UniformTSDFVolume.cpp
)

target_sources(tests PRIVATE
    odometry/Odometry.cpp
    odometry/OdometryOption.cpp
    odometry/OdometryTools.cpp
    odometry/RGBDOdometryJacobianFromColorTerm.cpp
    odometry/RGBDOdometryJacobianFromHybridTerm.cpp
)

target_sources(tests PRIVATE
    registration/CorrespondenceChecker.cpp
    registration/FastGlobalRegistration.cpp
    registration/Feature.cpp
    registration/GlobalOptimization.cpp
    registration/GlobalOptimizationConvergenceCriteria.cpp
    registration/PoseGraph.cpp
    registration/Registration.cpp
    registration/TransformationEstimation.cpp
)
