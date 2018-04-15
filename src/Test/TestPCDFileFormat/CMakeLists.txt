file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestPCDFileFormat ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(TestPCDFileFormat PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(TestPCDFileFormat Visualization IO Core)
set_target_properties(TestPCDFileFormat PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestPCDFileFormat
        RUNTIME DESTINATION test/bin)
