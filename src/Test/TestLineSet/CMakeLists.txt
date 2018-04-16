file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestLineSet ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(TestLineSet PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(TestLineSet Visualization IO Core)
set_target_properties(TestLineSet PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestLineSet
        RUNTIME DESTINATION test/bin)
