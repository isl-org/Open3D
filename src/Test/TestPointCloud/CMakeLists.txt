file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestPointCloud ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(TestPointCloud PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(TestPointCloud Visualization IO Core)
set_target_properties(TestPointCloud PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestPointCloud
        RUNTIME DESTINATION test/bin)
