add_subdirectory(ManuallyAlignPointCloud)

macro(open3d_add_tool TOOL_NAME)
    add_executable(${TOOL_NAME})

    target_sources(${TOOL_NAME} PRIVATE "${TOOL_NAME}.cpp")
    target_link_libraries(${TOOL_NAME} PRIVATE Open3D::Open3D ${ARGN})

    set_target_properties(${TOOL_NAME} PROPERTIES FOLDER "Tools")
    open3d_show_and_abort_on_warning(${TOOL_NAME})
    open3d_set_global_properties(${TOOL_NAME})
endmacro()

open3d_add_tool(ConvertPointCloud)
open3d_add_tool(GLInfo                  Open3D::3rdparty_opengl Open3D::3rdparty_glfw)
open3d_add_tool(ManuallyCropGeometry)
open3d_add_tool(MergeMesh)
open3d_add_tool(ViewGeometry)

add_executable(EmbedResources)
target_sources(EmbedResources PRIVATE EmbedResources.cpp)
set_target_properties(EmbedResources PROPERTIES FOLDER "Tools")
open3d_show_and_abort_on_warning(EmbedResources)
open3d_set_global_properties(EmbedResources)

add_executable(TestEmbedResources)
target_sources(TestEmbedResources PRIVATE TestEmbedResources.cpp)
target_link_libraries(TestEmbedResources PRIVATE Open3D::Open3D)
set_target_properties(TestEmbedResources PROPERTIES FOLDER "Tools")
add_dependencies(TestEmbedResources resource)
open3d_show_and_abort_on_warning(TestEmbedResources)
open3d_set_global_properties(TestEmbedResources)
