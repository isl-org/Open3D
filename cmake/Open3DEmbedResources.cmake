# open3d_embed_resources(<target>
#    OUTPUT_DIRECTORY <dir>
#    SOURCES <mat1> [<mat2>...]
# )
# Embeds the given set of resources into c++ source files that can then
# be compiled along with the rest of the library

function(open3d_embed_resources target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "OUTPUT_DIRECTORY" "SOURCES")

    # Check correct usage
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if (ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Missing values for arguments: ${ARG_KEYWORDS_MISSING_VALUES}")
    endif()

    if (NOT ARG_OUTPUT_DIRECTORY)
        message(FATAL_ERROR "No output directory for the embedded resource files specified.")
    endif()

    if (NOT ARG_SOURCES)
        message(FATAL_ERROR "No resource files specified for compilation.")
    endif()

set(RESOURCE_HEADER ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h)
set(RESOURCE_CPP ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp)

file(REMOVE ${RESOURCE_HEADER})
file(REMOVE ${RESOURCE_CPP})

foreach (resource IN LISTS ARG_SOURCES)
    # get_filename_component(RESOURCE_FULL_PATH "${resource}" ABSOLUTE)
    get_filename_component(RESOURCE_NAME "${resource}" NAME)

    string(REPLACE "." "_" EMBEDDED_RESOURCE_NAME ${RESOURCE_NAME})
    string(REPLACE "-" "_" EMBEDDED_RESOURCE_NAME ${EMBEDDED_RESOURCE_NAME})
    set(EMBEDDED_RESOURCE_FULL_PATH "${ARG_OUTPUT_DIRECTORY}/${EMBEDDED_RESOURCE_NAME}.cpp")

    file(RELATIVE_PATH EMBEDDED_RESOURCE_RELATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}" "${EMBEDDED_RESOURCE_FULL_PATH}")

    add_custom_command(
        OUTPUT
            ${EMBEDDED_RESOURCE_FULL_PATH}
        COMMAND
            EmbedResources -embed_gui_resource ${resource} ${ARG_OUTPUT_DIRECTORY}
        COMMENT
            "Building Embedded resource object ${EMBEDDED_RESOURCE_RELATIVE_PATH}"
        DEPENDS
            EmbedResources materials
        VERBATIM
    )

    # message(FATAL_ERROR ${ARG_OUTPUT_DIRECTORY})
    list(APPEND EMBEDDED_RESOURCES "${EMBEDDED_RESOURCE_FULL_PATH}")
endforeach()

add_custom_command(
    OUTPUT
        ${RESOURCE_HEADER}
        ${RESOURCE_CPP}
    DEPENDS
        EmbedResources
        ${EMBEDDED_RESOURCES}
    COMMAND
        EmbedResources -generate_resource_header ${RESOURCE_HEADER} ${ARG_OUTPUT_DIRECTORY}
    COMMENT
        "Generating Resource.h and Resource.cpp"
    VERBATIM
    )

add_custom_target(${target} ALL
    COMMAND
        echo "Embedding resources into the binary"
    DEPENDS
        ${RESOURCE_HEADER}
        ${RESOURCE_CPP}
        ${EMBEDDED_RESOURCES}
)

set_target_properties(${target} PROPERTIES EMBEDDED_RESOURCES "${EMBEDDED_RESOURCES}")

endfunction()
