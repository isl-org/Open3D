# open3d_embed_resources(<target>
#    OUTPUT_DIRECTORY <dir>
#    SOURCES <mat1> [<mat2>...]
# )

# --- build resources ----

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
# file(GLOB GUI_MATERIAL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Materials/*")

# foreach (material_src IN LISTS GUI_MATERIAL_SOURCE_FILES)
#     get_filename_component(MATERIAL_NAME "${material_src}" NAME)
#     string(REPLACE ".mat" "_filamat.cpp" EMBEDDED_MATERIAL_NAME ${MATERIAL_NAME})
#     set(EMBEDDED_MATERIAL_FULL_PATH "${OUTPUT_DIRECTORY}/${EMBEDDED_RESOURCE_NAME}.cpp")
#     list(APPEND EMBEDDED_MATERIALS "${EMBEDDED_MATERIAL_FULL_PATH}")
# endforeach()

# copy GUI/Resources -> <output>/resources

file(REMOVE ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h)
file(REMOVE ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp)


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
            EmbedResources ${resource} ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${ARG_OUTPUT_DIRECTORY} 
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
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp
    DEPENDS
        EmbedResources
        ${EMBEDDED_RESOURCES}
    COMMAND
        EmbedResources -complete ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h ${ARG_OUTPUT_DIRECTORY}
    COMMENT
        "Generating Resource.h and Resource.cpp"
    VERBATIM
    )

add_custom_target(${target} ALL
    COMMAND
        echo "Embedding resources into the binary"
    DEPENDS 
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp
        ${EMBEDDED_RESOURCES}
)



set(EMBEDDED_RESOURCES ${EMBEDDED_RESOURCES} PARENT_SCOPE)
# set_target_properties(${target} PROPERTIES COMPILED_RESOURCES "${COMPILED_RESOURCES}")

endfunction()
