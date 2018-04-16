file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(ManuallyAlignPointCloud ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(ManuallyAlignPointCloud PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(ManuallyAlignPointCloud Visualization IO Core)
set_target_properties(ManuallyAlignPointCloud PROPERTIES
        FOLDER "Tools"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS ManuallyAlignPointCloud
        RUNTIME DESTINATION bin)
