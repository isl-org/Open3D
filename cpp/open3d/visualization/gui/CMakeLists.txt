include(Open3DEmbedResources)
open3d_embed_resources(resource-embed)

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
    Resource.cpp
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

target_sources(GUI PRIVATE
    CompiledResources/brightday_ibl_ktx.cpp
    CompiledResources/brightday_skybox_ktx.cpp
    CompiledResources/colorMap_filamat.cpp
    CompiledResources/crossroads_ibl_ktx.cpp
    CompiledResources/crossroads_skybox_ktx.cpp
    CompiledResources/default_ibl_ktx.cpp
    CompiledResources/default_skybox_ktx.cpp
    CompiledResources/defaultGradient_png.cpp
    CompiledResources/defaultLit_filamat.cpp
    CompiledResources/defaultLitSSR_filamat.cpp
    CompiledResources/defaultLitTransparency_filamat.cpp
    CompiledResources/defaultTexture_png.cpp
    CompiledResources/defaultUnlit_filamat.cpp
    CompiledResources/defaultUnlitTransparency_filamat.cpp
    CompiledResources/depth_filamat.cpp
    CompiledResources/depth_value_filamat.cpp
    CompiledResources/hall_ibl_ktx.cpp
    CompiledResources/hall_skybox_ktx.cpp
    CompiledResources/img_blit_filamat.cpp
    CompiledResources/infiniteGroundPlane_filamat.cpp
    CompiledResources/konzerthaus_ibl_ktx.cpp
    CompiledResources/konzerthaus_skybox_ktx.cpp
    CompiledResources/nightlights_ibl_ktx.cpp
    CompiledResources/nightlights_skybox_ktx.cpp
    CompiledResources/normals_filamat.cpp
    CompiledResources/park_ibl_ktx.cpp
    CompiledResources/park_skybox_ktx.cpp
    CompiledResources/park2_ibl_ktx.cpp
    CompiledResources/park2_skybox_ktx.cpp
    CompiledResources/pillars_ibl_ktx.cpp
    CompiledResources/pillars_skybox_ktx.cpp
    CompiledResources/pointcloud_filamat.cpp
    CompiledResources/Roboto_Bold_ttf.cpp
    CompiledResources/Roboto_BoldItalic_ttf.cpp
    CompiledResources/Roboto_Medium_ttf.cpp
    CompiledResources/Roboto_MediumItalic_ttf.cpp
    CompiledResources/RobotoMono_Medium_ttf.cpp
    CompiledResources/streetlamp_ibl_ktx.cpp
    CompiledResources/streetlamp_skybox_ktx.cpp
    CompiledResources/ui_blit_filamat.cpp
    CompiledResources/unlitBackground_filamat.cpp
    CompiledResources/unlitGradient_filamat.cpp
    CompiledResources/unlitLine_filamat.cpp
    CompiledResources/unlitPolygonOffset_filamat.cpp
    CompiledResources/unlitSolidColor_filamat.cpp
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

open3d_show_and_abort_on_warning(GUI)
open3d_set_global_properties(GUI)
open3d_link_3rdparty_libraries(GUI)

# --- build resources ----
file(GLOB GUI_RESOURCE_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*")

# copy GUI/Resources -> <output>/resources
set(GUI_RESOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources")
file(MAKE_DIRECTORY ${GUI_RESOURCE_DIR})
file(COPY ${GUI_RESOURCE_SOURCE_FILES}
     DESTINATION ${GUI_RESOURCE_DIR})


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

add_dependencies(GUI materials resource-embed)

get_target_property(GUI_MATERIAL_COMPILED_FILES materials COMPILED_MATERIALS)

# Export GUI_RESOURCE_FILES to parent CMake context (cpp/open3d/)
set(GUI_RESOURCE_FILES
    ${GUI_RESOURCE_SOURCE_FILES} ${GUI_MATERIAL_COMPILED_FILES}
    PARENT_SCOPE)
set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)
