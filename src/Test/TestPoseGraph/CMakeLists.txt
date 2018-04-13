file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestPoseGraph ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(TestPoseGraph IO Core)
set_target_properties(TestPoseGraph PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestPoseGraph
        RUNTIME DESTINATION test/bin)
