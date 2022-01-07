# open3d_add_encoded_shader(<target>
#    OUTPUT_HEADER <header>
#    SOURCES <shader1> [<shader2>...]
# )
#
# Encodes the given set of shaders into a set of strings and stores them into <header>.
# The resulting header can be used in C++ code to pass the encoded shaders to the shader compiler.
#
# Use add_dependencies(<other_target> <target>) to build the header before <other_target> is built.
# Furthermore, make sure that <header> can be properly included in the C++ code.
function(open3d_add_encoded_shader target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "OUTPUT_HEADER" "SOURCES")

    # Check correct usage
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if (ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Missing values for arguments: ${ARG_KEYWORDS_MISSING_VALUES}")
    endif()

    if (NOT ARG_OUTPUT_HEADER)
        message(FATAL_ERROR "No output header file specified.")
    endif()

    if (NOT ARG_SOURCES)
        message(FATAL_ERROR "No shaders specified to generate the output header.")
    endif()

    # Build encoded shaders
    foreach (shader IN LISTS ARG_SOURCES)
        get_filename_component(SHADER_FULL_PATH "${shader}" ABSOLUTE)

        get_filename_component(SHADER_NAME "${shader}" NAME_WE)
        set(ENCODED_SHADER_FULL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.h")

        file(RELATIVE_PATH ENCODED_SHADER_RELATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}" "${ENCODED_SHADER_FULL_PATH}")

        add_custom_command(
            OUTPUT ${ENCODED_SHADER_FULL_PATH}
            COMMAND ShaderEncoder ${ENCODED_SHADER_FULL_PATH} ${SHADER_FULL_PATH}
            COMMENT "Building Encoded Shader object ${ENCODED_SHADER_RELATIVE_PATH}"
            MAIN_DEPENDENCY ${shader} DEPENDS ShaderEncoder
            VERBATIM
        )

        list(APPEND ENCODED_SHADERS "${ENCODED_SHADER_FULL_PATH}")
    endforeach()

    # Link encoded shaders
    get_filename_component(OUTPUT_FULL_PATH "${ARG_OUTPUT_HEADER}" ABSOLUTE)
    file(RELATIVE_PATH OUTPUT_RELATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}" "${OUTPUT_FULL_PATH}")

    add_custom_command(
        OUTPUT ${OUTPUT_FULL_PATH}
        COMMAND ShaderLinker ${OUTPUT_FULL_PATH} ${ENCODED_SHADERS}
        COMMENT "Linking Encoded Shader header ${OUTPUT_RELATIVE_PATH}"
        DEPENDS ${ENCODED_SHADERS} ShaderLinker
        VERBATIM
    )

    add_custom_target(${target} ALL
        DEPENDS "${OUTPUT_FULL_PATH}"
    )
endfunction()

# Helper target for open3d_add_encoded_shader
if (NOT TARGET ShaderEncoder)
    add_executable(ShaderEncoder EXCLUDE_FROM_ALL)
    target_sources(ShaderEncoder PRIVATE ${CMAKE_CURRENT_LIST_DIR}/ShaderEncoder.cpp)
    target_compile_features(ShaderEncoder PRIVATE cxx_std_14)
endif()

if (NOT TARGET ShaderLinker)
    add_executable(ShaderLinker EXCLUDE_FROM_ALL)
    target_sources(ShaderLinker PRIVATE ${CMAKE_CURRENT_LIST_DIR}/ShaderLinker.cpp)
    target_compile_features(ShaderLinker PRIVATE cxx_std_14)
endif()
