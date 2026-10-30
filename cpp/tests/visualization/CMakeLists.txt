if (BUILD_GUI)
    target_sources(tests PRIVATE
        rendering/MaterialModifier.cpp
        rendering/GaussianSplatRender.cpp
        rendering/Scene.cpp
    )
endif()
