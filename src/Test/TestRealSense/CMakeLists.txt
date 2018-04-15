file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestRealSense ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(TestRealSense PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(TestRealSense Visualization realsense)
set_target_properties(TestRealSense PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestRealSense
        RUNTIME DESTINATION test/bin)
