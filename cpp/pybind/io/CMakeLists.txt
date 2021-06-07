target_sources(pybind PRIVATE
    class_io.cpp
    io.cpp
)

if (BUILD_AZURE_KINECT)
    target_sources(pybind PRIVATE
        sensor.cpp
    )
endif()

if (BUILD_RPC_INTERFACE)
    target_sources(pybind PRIVATE
        rpc.cpp
    )
endif()
