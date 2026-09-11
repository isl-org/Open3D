# Apply a patch once to an ExternalProject source tree.
if(NOT PATCH_FILE OR NOT SOURCE_DIR)
    message(FATAL_ERROR "apply_patch.cmake: PATCH_FILE and SOURCE_DIR are required")
endif()

find_package(Git REQUIRED)

# git apply needs a repository context to perform its safety checks.
if(NOT EXISTS "${SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} init
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE git_init_result
        ERROR_VARIABLE git_init_error)
    if(NOT git_init_result EQUAL 0)
        message(FATAL_ERROR
            "apply_patch.cmake: failed to initialize ${SOURCE_DIR}: ${git_init_error}")
    endif()
endif()

execute_process(
    COMMAND ${GIT_EXECUTABLE} apply --reverse --check --ignore-whitespace "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE reverse_check_result
    OUTPUT_QUIET
    ERROR_QUIET)
if(reverse_check_result EQUAL 0)
    message(STATUS "Patch already applied, skipping: ${PATCH_FILE}")
    return()
endif()

execute_process(
    COMMAND ${GIT_EXECUTABLE} apply --ignore-whitespace --whitespace=fix "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE apply_result
    OUTPUT_VARIABLE apply_output
    ERROR_VARIABLE apply_error)
if(NOT apply_result EQUAL 0)
    message(FATAL_ERROR
        "apply_patch.cmake: failed to apply ${PATCH_FILE}\n"
        "Output: ${apply_output}\nError: ${apply_error}")
endif()
message(STATUS "Patch applied successfully: ${PATCH_FILE}")