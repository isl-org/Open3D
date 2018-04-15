file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestFileDialog ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(TestFileDialog PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(TestFileDialog Core IO tinyfiledialogs)
set_target_properties(TestFileDialog PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestFileDialog
        RUNTIME DESTINATION test/bin)
