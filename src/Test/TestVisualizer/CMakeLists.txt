file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestVisualizer ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(TestVisualizer PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(TestVisualizer Visualization IO Core)
set_target_properties(TestVisualizer PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestVisualizer
        RUNTIME DESTINATION test/bin)
