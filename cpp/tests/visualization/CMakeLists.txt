if (BUILD_GUI)
    target_sources(tests PRIVATE
        rendering/MaterialModifier.cpp
    )
endif()

add_executable(TestEmbedResources)
target_sources(TestEmbedResources PRIVATE TestEmbedResources.cpp)
target_link_libraries(TestEmbedResources PRIVATE Open3D::Open3D)
add_dependencies(TestEmbedResources resource-embed)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/test_embed)
add_custom_target(test-embed
    COMMAND
        echo "Running test for embedding resources" &&
        ./TestEmbedResources ${PROJECT_BINARY_DIR}/test_embed &&
        diff ${PROJECT_BINARY_DIR}/test_embed/brightday_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resources/brightday_ibl.ktx
    WORKING_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin/
    DEPENDS 
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp
)
add_dependencies(test-embed TestEmbedResources)
