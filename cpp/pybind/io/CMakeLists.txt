target_sources(pybind PRIVATE
    class_io.cpp
    io.cpp
    rpc.cpp
)

if (BUILD_AZURE_KINECT)
    target_sources(pybind PRIVATE
        sensor.cpp
    )
endif()
