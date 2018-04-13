file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(MergeMesh ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(MergeMesh IO Core)
set_target_properties(MergeMesh PROPERTIES
        FOLDER "Tools"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS MergeMesh
        RUNTIME DESTINATION bin)
