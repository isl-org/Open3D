file(GLOB_RECURSE CORE_ALL_SOURCE_FILES "*.cpp")

include_directories(".")
include_directories("/Camera")
include_directories("/Geometry")
include_directories("/Integration")
include_directories("/Odometry")
include_directories("/Registration")
include_directories("/Utility")

add_library(Core ${CORE_ALL_SOURCE_FILES})

target_link_libraries(Core ${JSONCPP_LIBRARIES})
