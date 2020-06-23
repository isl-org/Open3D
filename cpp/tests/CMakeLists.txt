
cmake_minimum_required(VERSION 3.0)
# Overide policy for hidden symbol visibility
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

include_directories(".")

file(GLOB_RECURSE UNIT_TEST_SOURCE_FILES "*.cpp")

# TODO: consider explicitly listing the files
if (NOT BUILD_AZURE_KINECT)
    set (EXCLUDE_DIR "io/sensor")
    foreach (TMP_PATH ${UNIT_TEST_SOURCE_FILES})
        string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
        if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
            list (REMOVE_ITEM UNIT_TEST_SOURCE_FILES ${TMP_PATH})
        endif ()
    endforeach(TMP_PATH)
endif()

# TODO: handle conditional compilation more systematically
if (NOT ENABLE_GUI)
    list(FILTER UNIT_TEST_SOURCE_FILES EXCLUDE REGEX .*/visualization/rendering/.*cpp)
endif()

add_executable(tests ${UNIT_TEST_SOURCE_FILES})
add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/test_data")

# If gpu not available, add "DISABLED_" to the gpu test names
if(BUILD_CUDA_MODULE)
    target_compile_definitions(tests PRIVATE GPU_CONDITIONAL_TEST_STR=) # Empty string
else()
    target_compile_definitions(tests PRIVATE GPU_CONDITIONAL_TEST_STR=DISABLED_)
endif()

find_package(Threads)
target_link_libraries(tests PRIVATE Threads::Threads ${CMAKE_PROJECT_NAME} ${JSONCPP_TARGET} ${GOOGLETEST_TARGET})
open3d_show_and_abort_on_warning(tests)
open3d_set_global_properties(tests)

if (BUILD_CUDA_MODULE)
    # We still need to explicitly link against CUDA libraries.
    # See: https://stackoverflow.com/a/48540499/1255535.
    # Consider removing dependencies of cuda headers in the future.
    target_link_libraries(tests PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(tests SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
