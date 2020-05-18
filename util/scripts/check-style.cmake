# ******************************************************************************
# Reference:
# https://github.com/NervanaSystems/ngraph/blob/master/cmake/Modules/style_check.cmake
#
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# Try to locate "yapf"
find_program(YAPF yapf PATHS ENV PATH)
if (YAPF)
    message(STATUS "yapf found at: ${YAPF}")
    execute_process(COMMAND ${YAPF} --version)
else()
    message(STATUS "Please Install YAPF (https://github.com/google/yapf)")
    message(STATUS "With PyPI:  `pip install yapf`")
    message(STATUS "With Conda: `conda install yapf`")
    message(FATAL_ERROR "yapf not found, python not available")
endif()

macro(style_check_file_python FILE)
    execute_process(
        COMMAND ${YAPF} --diff ${FILE}
        OUTPUT_VARIABLE STYLE_CHECK_RESULT
    )
    if (NOT STYLE_CHECK_RESULT STREQUAL "")
        message(STATUS "Style error: ${FILE}")
        list(APPEND ERROR_LIST_PYTHON ${FILE})
    endif()
endmacro()

set(DIRECTORIES_OF_INTEREST_PYTHON
    examples/Python
    src/UnitTest/Python
    docs
    src/Python
)

message(STATUS "Python check-style...")
foreach(DIRECTORY ${DIRECTORIES_OF_INTEREST_PYTHON})
    set(PY_GLOB "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.py")
    file(GLOB_RECURSE FILES ${PY_GLOB})
    foreach(FILE ${FILES})
        style_check_file_python(${FILE})
    endforeach(FILE)
endforeach(DIRECTORY)
if(ERROR_LIST_PYTHON)
    message(FATAL_ERROR "Style errors")
endif()
message(STATUS "Python check-style done")


# Try to locate "clang-format-5.0" and then "clang-format"
find_program(CLANG_FORMAT clang-format-5.0 PATHS ENV PATH)
if (NOT CLANG_FORMAT)
    find_program(CLANG_FORMAT clang-format PATHS ENV PATH)
endif()
if (CLANG_FORMAT)
    message(STATUS "clang-format found at: ${CLANG_FORMAT}")
    execute_process(COMMAND ${CLANG_FORMAT} --version)
else()
    message("See http://www.open3d.org/docs/release/contribute.html#automated-style-checker for help on style checker")
    message(FATAL_ERROR "clang-format not found, style not available")
endif()

macro(style_check_file_cpp FILE)
    execute_process(
        COMMAND ${CLANG_FORMAT} -style=file -output-replacements-xml ${FILE}
        OUTPUT_VARIABLE STYLE_CHECK_RESULT
    )
    if("${STYLE_CHECK_RESULT}" MATCHES ".*<replacement .*")
        message(STATUS "Style error: ${FILE}")
        list(APPEND ERROR_LIST_CPP ${FILE})
    endif()
endmacro()

set(DIRECTORIES_OF_INTEREST_CPP
    src
    examples
    docs/_static
)

message(STATUS "C++/CUDA check-style...")
foreach(DIRECTORY ${DIRECTORIES_OF_INTEREST_CPP})
    set(CPP_GLOB "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.cpp")
    set(CU_GLOB "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.cu")
    set(CUH_GLOB "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.cuh")
    set(H_GLOB "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.h")
    file(GLOB_RECURSE FILES ${CPP_GLOB} ${CU_GLOB} ${CUH_GLOB} ${H_GLOB})
    foreach(FILE ${FILES})
        style_check_file_cpp(${FILE})
    endforeach(FILE)
endforeach(DIRECTORY)
if(ERROR_LIST_CPP)
    message(FATAL_ERROR "Style errors")
endif()
message(STATUS "C++/CUDA check-style done")
