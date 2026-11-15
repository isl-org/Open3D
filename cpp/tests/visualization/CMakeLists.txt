if (BUILD_GUI)
    target_sources(tests PRIVATE
        rendering/Camera.cpp
        rendering/MaterialModifier.cpp
        rendering/GaussianSplatRender.cpp
    )
endif()
