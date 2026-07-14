# Platform-independent CMake patch execution script
if(NOT PATCH_FILE OR NOT SOURCE_DIR)
    message(FATAL_ERROR "PATCH_FILE and SOURCE_DIR must be specified!")
endif()

# Locate the Git executable
find_package(Git REQUIRED)

# Initialize a git repository context if one is not present
# (required by 'git apply' to enforce patch isolation and safety checks)
if(NOT EXISTS "${SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} init
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE git_init_res
        OUTPUT_QUIET
        ERROR_QUIET
    )
endif()

# Check if the patch is already applied (reverse check)
execute_process(
    COMMAND ${GIT_EXECUTABLE} apply --reverse --check --ignore-space-change --ignore-whitespace "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE res_reverse_check
    OUTPUT_QUIET
    ERROR_QUIET
)

if(res_reverse_check EQUAL 0)
    message(STATUS "Patch already applied, skipping: ${PATCH_FILE}")
    return()
endif()

# Apply the patch
execute_process(
    COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE res_apply
    OUTPUT_VARIABLE out_apply
    ERROR_VARIABLE err_apply
)

if(res_apply EQUAL 0)
    message(STATUS "Patch applied successfully: ${PATCH_FILE}")
else()
    message(WARNING "Failed to apply patch (may already be applied): ${PATCH_FILE}\nOutput: ${out_apply}\nError: ${err_apply}")
endif()
