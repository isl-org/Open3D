open3d_ispc_add_library(GUI OBJECT)

target_sources(GUI PRIVATE
    Application.cpp
    BitmapWindowSystem.cpp
    Button.cpp
    Checkbox.cpp
    Color.cpp
    ColorEdit.cpp
    Combobox.cpp
    Dialog.cpp
    Events.cpp
    FileDialog.cpp
    FileDialogNative.cpp
    Font.cpp
    GLFWWindowSystem.cpp
    Gui.cpp
    ImageWidget.cpp
    ImguiFilamentBridge.cpp
    Label.cpp
    Label3D.cpp
    Layout.cpp
    ListView.cpp
    Menu.cpp
    MenuImgui.cpp
    NumberEdit.cpp
    RadioButton.cpp
    PickPointsInteractor.cpp
    ProgressBar.cpp
    SceneWidget.cpp
    Slider.cpp
    StackedWidget.cpp
    TabControl.cpp
    Task.cpp
    TextEdit.cpp
    Theme.cpp
    ToggleSwitch.cpp
    TreeView.cpp
    UIImage.cpp
    Util.cpp
    VectorEdit.cpp
    Widget.cpp
    WidgetProxy.cpp
    WidgetStack.cpp
    Window.cpp
)

if (WIN32)
    target_sources(GUI PRIVATE
        NativeWin32.cpp
    )
elseif (APPLE)
    target_sources(GUI PRIVATE
        NativeMacOS.mm
        MenuMacOS.mm
    )
else()
    target_sources(GUI PRIVATE
        NativeLinux.cpp
    )
endif()
# filament uses C++20 designated initializers
set_property(TARGET GUI PROPERTY CXX_STANDARD 20)


open3d_show_and_abort_on_warning(GUI)
open3d_set_global_properties(GUI)
open3d_link_3rdparty_libraries(GUI)

# --- build resources ----
file(GLOB GUI_RESOURCE_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*")

set(GAUSSIAN_COMPUTE_SHADER_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_project.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_prefix_sum.comp
    # Subgroup variant: used on Linux/macOS where subgroup arithmetic is reliable.
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_prefix_sum_subgroup.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_scatter.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_composite.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_radix_sort_histograms.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_radix_sort.comp
    # Subgroup variant: used on Linux/macOS where subgroup arithmetic is reliable.
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_radix_sort_subgroup.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_radix_sort_keygen.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_radix_sort_payload.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_compute_dispatch_args.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_depth_merge.comp
    # Test-only Vulkan compute probe for device forward progress.
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_forward_progress_chain.comp
    # OneSweep sort: subgroup-only variants; fall back to radix sort if unavailable.
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_onesweep_global_hist_subgroup.comp
    ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/rendering/gaussian_splat/shaders/gaussian_onesweep_digit_pass_subgroup.comp
)

# copy GUI/Resources -> <output>/resources
set(GUI_RESOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources")
file(MAKE_DIRECTORY ${GUI_RESOURCE_DIR})
file(COPY ${GUI_RESOURCE_SOURCE_FILES}
     DESTINATION ${GUI_RESOURCE_DIR})

include(Open3DAddComputeShaders)

open3d_add_compute_shaders(gaussian_compute_shaders
    OUTPUT_DIRECTORY
    ${GUI_RESOURCE_DIR}/gaussian_splat
    SOURCES
    ${GAUSSIAN_COMPUTE_SHADER_SOURCE_FILES}
)


include(Open3DAddCompiledMaterials)

open3d_add_compiled_materials(materials
    OUTPUT_DIRECTORY
    ${GUI_RESOURCE_DIR}
    SOURCES
    Materials/colorMap.mat
    Materials/defaultLit.mat
    Materials/defaultLitSSR.mat
    Materials/defaultLitTransparency.mat
    Materials/defaultUnlit.mat
    Materials/defaultUnlitTransparency.mat
    Materials/depth_value.mat
    Materials/depth.mat
    Materials/img_blit.mat
    Materials/infiniteGroundPlane.mat
    Materials/normals.mat
    Materials/pointcloud.mat
    Materials/ui_blit.mat
    Materials/unlitBackground.mat
    Materials/unlitGradient.mat
    Materials/unlitLine.mat
    Materials/unlitPolygonOffset.mat
    Materials/unlitSolidColor.mat
)

# Source group for Visual Studio
add_source_group(Materials)

add_dependencies(GUI materials)
add_dependencies(GUI gaussian_compute_shaders)

get_target_property(GUI_MATERIAL_COMPILED_FILES materials COMPILED_MATERIALS)
get_target_property(GAUSSIAN_COMPUTE_SHADER_OUTPUT_FILES gaussian_compute_shaders COMPUTE_SHADER_FILES)

# Export GUI_RESOURCE_FILES to parent CMake context (cpp/open3d/)
set(GUI_RESOURCE_FILES
    ${GUI_RESOURCE_SOURCE_FILES} ${GUI_MATERIAL_COMPILED_FILES}
    ${GAUSSIAN_COMPUTE_SHADER_OUTPUT_FILES}
    PARENT_SCOPE)
set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)
