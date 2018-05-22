# build
file(GLOB_RECURSE CORE_ALL_SOURCE_FILES "*.cpp")

include_directories(".")
include_directories("/Camera")
include_directories("/Geometry")
include_directories("/Integration")
include_directories("/Odometry")
include_directories("/Registration")
include_directories("/Utility")

# create object library
add_library(Core OBJECT ${CORE_ALL_SOURCE_FILES})

#install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)
