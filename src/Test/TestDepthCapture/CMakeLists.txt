file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestDepthCapture ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(TestDepthCapture Visualization IO Core)
set_target_properties(TestDepthCapture PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestDepthCapture
        RUNTIME DESTINATION test/bin)
