if (BUILD_GUI)
    target_sources(tests PRIVATE
        rendering/MaterialModifier.cpp
        rendering/GaussianSplatRender.cpp
    )
endif()

if (BUILD_GUI)
    target_sources(tests PRIVATE rendering/TensorLineSetUpdate.cpp)
endif()
