file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(ConvertPointCloud ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ConvertPointCloud IO Core)
set_target_properties(ConvertPointCloud PROPERTIES
        FOLDER "Tools"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS ConvertPointCloud
        RUNTIME DESTINATION bin)
