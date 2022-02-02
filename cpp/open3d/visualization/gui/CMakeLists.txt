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

add_dependencies(GUI materials)

get_target_property(GUI_MATERIAL_COMPILED_FILES materials COMPILED_MATERIALS)

# Export GUI_RESOURCE_FILES to parent CMake context (cpp/open3d/)
set(GUI_RESOURCE_FILES
    ${GUI_RESOURCE_SOURCE_FILES} ${GUI_MATERIAL_COMPILED_FILES}
    PARENT_SCOPE)
set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)
