file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
project(TestTriangleMesh)
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Visualization IO Core)
set_target_properties(${PROJECT_NAME} PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION test/bin)
