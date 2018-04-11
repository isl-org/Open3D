file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestProgramOptions ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(TestProgramOptions Core)
set_target_properties(TestProgramOptions PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestProgramOptions
        RUNTIME DESTINATION test/bin)
