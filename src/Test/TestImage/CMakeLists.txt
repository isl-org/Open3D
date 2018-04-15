file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestImage ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(TestImage PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(TestImage IO Core)
set_target_properties(TestImage PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestImage
        RUNTIME DESTINATION test/bin)
