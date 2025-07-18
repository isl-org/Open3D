# Check (and apply) style for C++/CUDA files.
#
# This cmake file has the same functionality as as the CppFormater in
# check_style.py, but works without any python dependencies.
#
# This cmake file expects the following variables:
# > PROJECT_SOURCE_DIR
# > APPLY
#
# Example usage:
# cmake -DPROJECT_SOURCE_DIR=/path/to/open3d \
#       -DAPPLY=ON \
#       -P check_cpp_style.cmake

option(APPLY "Apply style to files in-place." OFF)

find_program(CLANG_FORMAT clang-format PATHS ENV PATH)
if(CLANG_FORMAT)
    message(STATUS "clang-format found at: ${CLANG_FORMAT}")
    execute_process(COMMAND ${CLANG_FORMAT} --version)
else()
    message("See https://www.open3d.org/docs/release/contribute/styleguide.html#style-guide for help on style checker")
    message(FATAL_ERROR "clang-format not found, style not available")
endif()

# Process individual file
macro(style_apply_file_cpp FILE)
    execute_process(
        COMMAND ${CLANG_FORMAT} -style=file -output-replacements-xml ${FILE}
        OUTPUT_VARIABLE STYLE_CHECK_RESULT
    )
    if("${STYLE_CHECK_RESULT}" MATCHES ".*<replacement .*")
        if(APPLY)
            message(STATUS "Style applied for: ${FILE}")
            execute_process(COMMAND ${CLANG_FORMAT} -style=file -i ${FILE})
        else()
            message(STATUS "Style error: ${FILE}")
            list(APPEND ERROR_LIST_CPP ${FILE})
        endif()
    endif()
endmacro()

# Note: also modify CPP_FORMAT_DIRS in check_style.py.
set(CPP_FORMAT_DIRS
    cpp
    examples
    docs/_static
)

if(APPLY)
    message(STATUS "C++/CUDA apply-style...")
else()
    message(STATUS "C++/CUDA check-style...")
endif()
foreach(DIRECTORY ${CPP_FORMAT_DIRS})
    file(GLOB_RECURSE FILES
        # C++
        "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.h"
        "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.cpp"
        # CUDA
        "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.cuh"
        "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.cu"
        # ISPC
        "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.isph"
        "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.ispc"
        # Generated files
        "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.h.in"
    )
    set(IGNOFRED_FILES
        "${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/shader/Shader.h"
    )
    list(REMOVE_ITEM FILES ${IGNOFRED_FILES})

    foreach(FILE ${FILES})
        style_apply_file_cpp(${FILE})
    endforeach(FILE)
endforeach(DIRECTORY)
if(APPLY)
    message(STATUS "C++/CUDA apply-style done.")
else()
    message(STATUS "C++/CUDA check-style done.")
endif()

# Throw error if under style check mode.
if(ERROR_LIST_CPP AND (NOT APPLY))
    message(FATAL_ERROR "Style errors found.")
endif()
