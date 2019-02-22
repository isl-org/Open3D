# build
file(GLOB_RECURSE CORE_ALL_SOURCE_FILES
    "Camera/*.cpp"
    "ColorMap/*.cpp"
    "Geometry/*.cpp"
    "Integration/*.cpp"
    "Odometry/*.cpp"
    "Registration/*.cpp"
    "Utility/*.cpp"
)
file(GLOB_RECURSE CORE_ALL_HEADER_FILES
    "Camera/*.h"
    "ColorMap/*.h"
    "Geometry/*.h"
    "Integration/*.h"
    "Odometry/*.h"
    "Registration/*.h"
    "Utility/*.h"
)

# create object library
add_library(Core OBJECT ${CORE_ALL_HEADER_FILES} ${CORE_ALL_SOURCE_FILES})

# create subgroups
ADD_SOURCE_GROUP(Camera)
ADD_SOURCE_GROUP(ColorMap)
ADD_SOURCE_GROUP(Geometry)
ADD_SOURCE_GROUP(Integration)
ADD_SOURCE_GROUP(Odometry)
ADD_SOURCE_GROUP(Registration)
ADD_SOURCE_GROUP(Utility)

#install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)
