
cmake_minimum_required(VERSION 3.0)

include_directories(".")

file(GLOB_RECURSE UNIT_TEST_SOURCE_FILES "*.cpp")

# TODO: consider explicitly listing the files
if (NOT BUILD_AZURE_KINECT)
    set (EXCLUDE_DIR "IO/Sensor")
    foreach (TMP_PATH ${UNIT_TEST_SOURCE_FILES})
        string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
        if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
            list (REMOVE_ITEM UNIT_TEST_SOURCE_FILES ${TMP_PATH})
        endif ()
    endforeach(TMP_PATH)
endif()

add_executable(unitTests ${UNIT_TEST_SOURCE_FILES})
add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/TestData")

target_link_libraries(unitTests pthread ${CMAKE_PROJECT_NAME})
ShowAndAbortOnWarning(unitTests)
