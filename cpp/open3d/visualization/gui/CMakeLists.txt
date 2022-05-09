add_custom_command(
    OUTPUT
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp
        ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h
    DEPENDS
        EmbedResources
        materials
    COMMAND
        ${CMAKE_COMMAND} -E rm -f ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp &&
        ${CMAKE_COMMAND} -E rm -f ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.h 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/brightday_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/brightday_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/colorMap.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/crossroads_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/crossroads_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultGradient.png ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/default_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultLit.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultLitSSR.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultLitTransparency.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/default_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultTexture.png ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultUnlit.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/defaultUnlitTransparency.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/depth.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/depth_value.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/hall_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/hall_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/img_blit.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/infiniteGroundPlane.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/konzerthaus_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/konzerthaus_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/nightlights_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/nightlights_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/normals.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/park2_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/park2_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/park_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/park_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/pillars_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/pillars_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/pointcloud.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/Roboto-Bold.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/Roboto-BoldItalic.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/Roboto-Medium.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/Roboto-MediumItalic.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/RobotoMono-Medium.ttf ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/streetlamp_ibl.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/streetlamp_skybox.ktx ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/ui_blit.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitBackground.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitGradient.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitLine.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitPolygonOffset.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources ${PROJECT_BINARY_DIR}/bin/resources/unlitSolidColor.filamat ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp 
    COMMAND
        EmbedResources -complete ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp
    COMMENT
        "Generating Resource.cpp"
    VERBATIM
    )

add_custom_target(resource-embed ALL
    COMMAND
        echo "Embedding resources into the binary"
    DEPENDS ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/gui/Resource.cpp
)

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
