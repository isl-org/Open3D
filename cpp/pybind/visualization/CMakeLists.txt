target_sources(pybind PRIVATE
    renderoption.cpp
    utility.cpp
    viewcontrol.cpp
    visualization.cpp
    visualizer.cpp
    rendering/material.cpp  # for RPC serialization
)

if (BUILD_GUI)
    target_sources(pybind PRIVATE
        o3dvisualizer.cpp
    )

    target_sources(pybind PRIVATE
        app/viewer.cpp
    )

    target_sources(pybind PRIVATE
        gui/events.cpp
        gui/gui.cpp
    )

    target_sources(pybind PRIVATE
        rendering/rendering.cpp
    )
endif()

if (BUILD_WEBRTC)
    target_sources(pybind PRIVATE
        webrtc_server/webrtc_window_system.cpp
    )
endif()
