file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestOpenMP ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(TestOpenMP Core)
set_target_properties(TestOpenMP PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestOpenMP
        RUNTIME DESTINATION test/bin)
