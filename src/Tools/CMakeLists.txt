add_subdirectory(ManuallyAlignPointCloud)

macro(TOOL TOOL_NAME)
    add_executable(${TOOL_NAME} "${TOOL_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${TOOL_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${TOOL_NAME} PROPERTIES FOLDER "Tools")
endmacro(TOOL)

TOOL(ConvertPointCloud      Core IO)
TOOL(EncodeShader)
TOOL(ManuallyCropPointCloud Core IO Visualization)
TOOL(MergeMesh              Core IO)
TOOL(ViewGeometry           Core IO Visualization)
