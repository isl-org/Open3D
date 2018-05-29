# build
file(GLOB_RECURSE CORE_ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Core OBJECT ${CORE_ALL_SOURCE_FILES})

set(CORE_INCLUDE_DIRS "")
Directories(${CMAKE_CURRENT_SOURCE_DIR} CORE_INCLUDE_DIRS)
target_include_directories(Core PUBLIC ${CORE_INCLUDE_DIRS})

message("")
foreach(child ${CORE_INCLUDE_DIRS})
    message("${child}")
endforeach()
message("")

#install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)
