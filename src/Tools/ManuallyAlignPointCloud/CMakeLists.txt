file(GLOB SOURCE_FILES "*.cpp")

include_directories(".")

add_executable(ManuallyAlignPointCloud ${SOURCE_FILES})

target_link_libraries(ManuallyAlignPointCloud Open3D)

set_target_properties(ManuallyAlignPointCloud PROPERTIES FOLDER "Tools")
