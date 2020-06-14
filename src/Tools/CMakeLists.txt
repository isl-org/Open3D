add_subdirectory(ManuallyAlignPointCloud)

macro(TOOL TOOL_NAME)
    add_executable(${TOOL_NAME} "${TOOL_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${TOOL_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${TOOL_NAME} PROPERTIES FOLDER "Tools")
    open3d_show_and_abort_on_warning(${TOOL_NAME})
    open3d_set_global_properties(${TOOL_NAME})
endmacro(TOOL)

TOOL(ConvertPointCloud      ${PROJECT_NAME})
TOOL(EncodeShader)
TOOL(GLInfo                 ${PROJECT_NAME} ${GLFW_TARGET} ${OPENGL_TARGET})
TOOL(ManuallyCropGeometry   ${PROJECT_NAME})
TOOL(MergeMesh              ${PROJECT_NAME})
TOOL(ViewGeometry           ${PROJECT_NAME})
