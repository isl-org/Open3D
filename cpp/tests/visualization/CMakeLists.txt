if (BUILD_GUI)
    target_sources(tests PRIVATE
        rendering/MaterialModifier.cpp
    )
endif()


add_executable(TreeViewTest
    TreeViewTest.cpp
)

target_link_libraries(TreeViewTest
    PRIVATE
        Open3D::Open3D
)

set_target_properties(TreeViewTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
