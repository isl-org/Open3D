target_sources(tests PRIVATE
    TransformationConverter.cpp
)

target_sources(tests PRIVATE
    odometry/RGBDOdometry.cpp
)

target_sources(tests PRIVATE
    registration/Feature.cpp
    registration/Registration.cpp
    registration/TransformationEstimation.cpp
)

target_sources(tests PRIVATE
    slac/ControlGrid.cpp
    slac/SLAC.cpp
)
