file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(ManuallyCropPointCloud ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ManuallyCropPointCloud Visualization IO Core)
set_target_properties(ManuallyCropPointCloud PROPERTIES
        FOLDER "Tools"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS ManuallyCropPointCloud
        RUNTIME DESTINATION bin)
