file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")

include_directories(".")

add_executable(ManuallyAlignPointCloud ${HEADER_FILES} ${SOURCE_FILES})
open3d_show_and_abort_on_warning(ManuallyAlignPointCloud)

target_link_libraries(ManuallyAlignPointCloud PRIVATE ${PROJECT_NAME} ${JSONCPP_TARGET} ${TINYFILEDIALOGS_TARGET})

set_target_properties(ManuallyAlignPointCloud PROPERTIES FOLDER "Tools")
