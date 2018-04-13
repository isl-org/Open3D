file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestTriangleMesh ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(TestTriangleMesh Visualization IO Core)
set_target_properties(TestTriangleMesh PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestTriangleMesh
        RUNTIME DESTINATION test/bin)
