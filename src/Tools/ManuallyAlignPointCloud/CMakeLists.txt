file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")

include_directories(".")

add_executable(ManuallyAlignPointCloud ${HEADER_FILES} ${SOURCE_FILES})
ShowAndAbortOnWarning(ManuallyAlignPointCloud)

target_link_libraries(ManuallyAlignPointCloud ${CMAKE_PROJECT_NAME})

set_target_properties(ManuallyAlignPointCloud PROPERTIES FOLDER "Tools")
