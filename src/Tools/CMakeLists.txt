add_subdirectory(ManuallyAlignPointCloud)

macro(TOOL TOOL_NAME)
    add_executable(${TOOL_NAME} "${TOOL_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${TOOL_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${TOOL_NAME} PROPERTIES FOLDER "Tools")
    ShowAndAbortOnWarning(${TOOL_NAME})
endmacro(TOOL)

TOOL(ConvertPointCloud      ${CMAKE_PROJECT_NAME})
TOOL(EncodeShader)
TOOL(ManuallyCropGeometry   ${CMAKE_PROJECT_NAME})
TOOL(MergeMesh              ${CMAKE_PROJECT_NAME})
TOOL(ViewGeometry           ${CMAKE_PROJECT_NAME})
