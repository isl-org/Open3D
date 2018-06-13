# build
file(GLOB_RECURSE CORE_ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Core OBJECT ${CORE_ALL_SOURCE_FILES})

Directories(${CMAKE_CURRENT_SOURCE_DIR} CORE_INCLUDE_DIRS)
set(CORE_INCLUDE_DIRS ${CORE_INCLUDE_DIRS} PARENT_SCOPE)

target_include_directories(Core PUBLIC ${CORE_INCLUDE_DIRS})

#install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)
