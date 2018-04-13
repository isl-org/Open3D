file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(ViewGeometry ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ViewGeometry Visualization IO Core)
set_target_properties(ViewGeometry PROPERTIES
        FOLDER "Tools"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS ViewGeometry
        RUNTIME DESTINATION bin)
