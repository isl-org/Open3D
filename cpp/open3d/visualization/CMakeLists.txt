open3d_ispc_add_library(visualization OBJECT)

open3d_ispc_add_library(visualization_impl OBJECT)
set_target_properties(visualization_impl PROPERTIES CXX_VISIBILITY_PRESET hidden)
# filament uses C++20 designated initializers
set_property(TARGET visualization_impl visualization PROPERTY CXX_STANDARD 20)

target_sources(visualization_impl PRIVATE
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

target_sources(visualization PRIVATE
    rendering/Material.cpp  # For RPC serialization
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
        rendering/filament/FilamentEngine.cpp
        rendering/filament/FilamentRenderer.cpp
    )

    target_sources(visualization_impl PRIVATE
        rendering/filament/FilamentCamera.cpp
        rendering/filament/FilamentEntitiesMods.cpp
        rendering/filament/FilamentGeometryBuffersBuilder.cpp
        rendering/filament/FilamentRenderToBuffer.cpp
        rendering/filament/FilamentResourceManager.cpp
        rendering/filament/FilamentScene.cpp
        rendering/filament/FilamentView.cpp
        rendering/filament/LineSetBuffers.cpp
        rendering/filament/PointCloudBuffers.cpp
        rendering/filament/TriangleMeshBuffers.cpp
        rendering/filament/GaussianSplatBuffers.cpp
    )

    target_sources(visualization PRIVATE
        utility/Draw.cpp
    )

    target_sources(visualization PRIVATE
        visualizer/MessageProcessor.cpp
        visualizer/GuiSettingsModel.cpp
        visualizer/GuiSettingsView.cpp
        visualizer/GuiVisualizer.cpp
        visualizer/GuiWidgets.cpp
        visualizer/O3DVisualizer.cpp
        visualizer/O3DVisualizerSelections.cpp
    )

    target_sources(visualization PRIVATE
        app/Viewer.cpp
    )
endif()


include(Open3DAddEncodedShader)

open3d_add_encoded_shader(shader
    OUTPUT_HEADER
    shader/Shader.h
    SOURCES
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

# Source group for Visual Studio
add_source_group(shader/glsl)

add_dependencies(visualization_impl shader)


open3d_show_and_abort_on_warning(visualization)
open3d_set_global_properties(visualization)
open3d_set_open3d_lib_properties(visualization)
open3d_link_3rdparty_libraries(visualization)

open3d_show_and_abort_on_warning(visualization_impl)
open3d_set_global_properties(visualization_impl)
open3d_set_open3d_lib_properties(visualization_impl)
open3d_link_3rdparty_libraries(visualization_impl)

# export GUI_RESOURCE_DIR to parent scope
if (BUILD_GUI)
    set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)
endif()
