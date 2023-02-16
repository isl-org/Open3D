target_sources(pybind PRIVATE
    pipelines.cpp
)

target_sources(pybind PRIVATE
    odometry/odometry.cpp
)

target_sources(pybind PRIVATE
    registration/feature.cpp
    registration/registration.cpp
    registration/robust_kernel.cpp
)

target_sources(pybind PRIVATE
    slac/slac.cpp
)

target_sources(pybind PRIVATE
    slam/slam.cpp
)
