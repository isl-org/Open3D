add_library(visualization OBJECT)

target_sources(visualization PRIVATE
    shader/GeometryRenderer.cpp
    shader/ImageMaskShader.cpp
    shader/ImageShader.cpp
    shader/NormalShader.cpp
    shader/PhongShader.cpp
    shader/PickingShader.cpp
    shader/RGBDImageShader.cpp
    shader/ShaderWrapper.cpp
    shader/Simple2DShader.cpp
    shader/SimpleBlackShader.cpp
    shader/SimpleShader.cpp
    shader/TexturePhongShader.cpp
    shader/TextureSimpleShader.cpp
)

target_sources(visualization PRIVATE
    utility/ColorMap.cpp
    utility/DrawGeometry.cpp
    utility/GLHelper.cpp
    utility/PointCloudPicker.cpp
    utility/SelectionPolygon.cpp
    utility/SelectionPolygonVolume.cpp
)

target_sources(visualization PRIVATE
    visualizer/RenderOption.cpp
    visualizer/RenderOptionWithEditing.cpp
    visualizer/ViewControl.cpp
    visualizer/ViewControlWithCustomAnimation.cpp
    visualizer/ViewControlWithEditing.cpp
    visualizer/ViewParameters.cpp
    visualizer/ViewTrajectory.cpp
    visualizer/Visualizer.cpp
    visualizer/VisualizerCallback.cpp
    visualizer/VisualizerRender.cpp
    visualizer/VisualizerWithCustomAnimation.cpp
    visualizer/VisualizerWithEditing.cpp
    visualizer/VisualizerWithKeyCallback.cpp
    visualizer/VisualizerWithVertexSelection.cpp
)

if (BUILD_GUI)
    target_sources(visualization PRIVATE
        rendering/Camera.cpp
        rendering/CameraInteractorLogic.cpp
        rendering/CameraSphereInteractorLogic.cpp
        rendering/ColorGrading.cpp
        rendering/Gradient.cpp
        rendering/IBLRotationInteractorLogic.cpp
        rendering/LightDirectionInteractorLogic.cpp
        rendering/MaterialModifier.cpp
        rendering/MatrixInteractorLogic.cpp
        rendering/ModelInteractorLogic.cpp
        rendering/Open3DScene.cpp
        rendering/Renderer.cpp
        rendering/RendererHandle.cpp
        rendering/RotationInteractorLogic.cpp
    )

    target_sources(visualization PRIVATE
        rendering/filament/FilamentCamera.cpp
        rendering/filament/FilamentEngine.cpp
        rendering/filament/FilamentEntitiesMods.cpp
        rendering/filament/FilamentGeometryBuffersBuilder.cpp
        rendering/filament/FilamentRenderer.cpp
        rendering/filament/FilamentRenderToBuffer.cpp
        rendering/filament/FilamentResourceManager.cpp
        rendering/filament/FilamentScene.cpp
        rendering/filament/FilamentView.cpp
        rendering/filament/LineSetBuffers.cpp
        rendering/filament/PointCloudBuffers.cpp
        rendering/filament/TriangleMeshBuffers.cpp
    )

    target_sources(visualization PRIVATE
        utility/Draw.cpp
    )

    target_sources(visualization PRIVATE
        visualizer/GuiSettingsModel.cpp
        visualizer/GuiSettingsView.cpp
        visualizer/GuiVisualizer.cpp
        visualizer/GuiWidgets.cpp
        visualizer/O3DVisualizer.cpp
        visualizer/O3DVisualizerSelections.cpp
    )
endif()

if (BUILD_RPC_INTERFACE)
    target_sources(visualization PRIVATE
        visualizer/Receiver.cpp
    )
endif()


# some black magic to automatically create a header file from shader files
set(SHADER_FILES
    shader/glsl/ImageFragmentShader.glsl
    shader/glsl/ImageMaskFragmentShader.glsl
    shader/glsl/ImageMaskVertexShader.glsl
    shader/glsl/ImageVertexShader.glsl
    shader/glsl/NormalFragmentShader.glsl
    shader/glsl/NormalVertexShader.glsl
    shader/glsl/PhongFragmentShader.glsl
    shader/glsl/PhongVertexShader.glsl
    shader/glsl/PickingFragmentShader.glsl
    shader/glsl/PickingVertexShader.glsl
    shader/glsl/RGBDImageFragmentShader.glsl
    shader/glsl/Simple2DFragmentShader.glsl
    shader/glsl/Simple2DVertexShader.glsl
    shader/glsl/SimpleBlackFragmentShader.glsl
    shader/glsl/SimpleBlackVertexShader.glsl
    shader/glsl/SimpleFragmentShader.glsl
    shader/glsl/SimpleVertexShader.glsl
    shader/glsl/TexturePhongFragmentShader.glsl
    shader/glsl/TexturePhongVertexShader.glsl
    shader/glsl/TextureSimpleFragmentShader.glsl
    shader/glsl/TextureSimpleVertexShader.glsl
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/shader/Shader.h"
    COMMAND EncodeShader "${CMAKE_CURRENT_SOURCE_DIR}/shader/Shader.h"
    DEPENDS ${SHADER_FILES} EncodeShader
)

foreach(SHADER ${SHADER_FILES})
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/shader/Shader.h"
        COMMAND EncodeShader "${CMAKE_CURRENT_SOURCE_DIR}/shader/Shader.h" "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}"
        DEPENDS ${SHADER_FILES} EncodeShader
        APPEND
    )
endforeach()

add_custom_target(shader ALL
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shader/Shader.h"
)
set_target_properties(shader PROPERTIES FOLDER "Custom")

source_group("Source Files\\Shader\\GLSL" FILES ${SHADER_FILES})

add_dependencies(visualization shader)


open3d_show_and_abort_on_warning(visualization)
open3d_set_global_properties(visualization)
open3d_set_open3d_lib_properties(visualization)
open3d_link_3rdparty_libraries(visualization)

# export GUI_RESOURCE_DIR to parent scope
if (BUILD_GUI)
    set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)
endif()
