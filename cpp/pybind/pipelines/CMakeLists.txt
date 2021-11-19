target_sources(pybind PRIVATE
    pipelines.cpp
)

target_sources(pybind PRIVATE
    color_map/color_map.cpp
)

target_sources(pybind PRIVATE
    integration/integration.cpp
)

target_sources(pybind PRIVATE
    odometry/odometry.cpp
)

target_sources(pybind PRIVATE
    registration/feature.cpp
    registration/global_optimization.cpp
    registration/registration.cpp
    registration/robust_kernels.cpp
)
