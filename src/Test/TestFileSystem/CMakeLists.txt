file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestFileSystem ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(TestFileSystem PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(TestFileSystem IO Core)
set_target_properties(TestFileSystem PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestFileSystem
        RUNTIME DESTINATION test/bin)
