file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestCameraPoseTrajectory ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(TestCameraPoseTrajectory Visualization IO Core)
set_target_properties(TestCameraPoseTrajectory PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestCameraPoseTrajectory
        RUNTIME DESTINATION test/bin)
