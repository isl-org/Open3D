# ******************************************************************************
# Reference:
# https://github.com/NervanaSystems/ngraph/blob/master/cmake/Modules/style_apply.cmake
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

# Tries to locate "clang-format-5.0" and then "clang-format"
find_program(CLANG_FORMAT clang-format-5.0 PATHS ENV PATH)
if (NOT CLANG_FORMAT)
    find_program(CLANG_FORMAT clang-format PATHS ENV PATH)
endif()
if (CLANG_FORMAT)
    message(STATUS "clang-format found at: ${CLANG_FORMAT}")
    execute_process(COMMAND ${CLANG_FORMAT} --version)
else()
    message("See http://www.open3d.org/docs/contribute.html#automated-style-checker for help on style checker")
    message(FATAL_ERROR "clang-format not found, style not available")
endif()

function(style_apply_file PATH)
    execute_process(
        COMMAND ${CLANG_FORMAT} -style=file -output-replacements-xml ${FILE}
        OUTPUT_VARIABLE STYLE_CHECK_RESULT
    )
    if("${STYLE_CHECK_RESULT}" MATCHES ".*<replacement .*")
        message(STATUS "Style applied for: ${FILE}")
        execute_process(COMMAND ${CLANG_FORMAT} -style=file -i ${PATH})
    endif()

endfunction()

set(DIRECTORIES_OF_INTEREST
    src
    examples
    docs/_static
)

foreach(DIRECTORY ${DIRECTORIES_OF_INTEREST})
    set(CPP_GLOB "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.cpp")
    set(H_GLOB "${PROJECT_SOURCE_DIR}/${DIRECTORY}/*.h")
    file(GLOB_RECURSE FILES ${CPP_GLOB} ${H_GLOB})
    foreach(FILE ${FILES})
        style_apply_file(${FILE})
    endforeach(FILE)
endforeach(DIRECTORY)
message(STATUS "apply-style done")
