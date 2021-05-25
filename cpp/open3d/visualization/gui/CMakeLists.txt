add_library(GUI OBJECT)

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

set(GUI_MATERIAL_SOURCE_FILES
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

# copy GUI/Resources -> <output>/resources
set(GUI_RESOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources")
file(MAKE_DIRECTORY ${GUI_RESOURCE_DIR})
file(COPY ${GUI_RESOURCE_SOURCE_FILES}
     DESTINATION ${GUI_RESOURCE_DIR})

# Compile .mat files and copy to resources directory in output directory
if (IOS OR ANDROID)
    set(GUI_MATC_ARGS --platform mobile)
elseif (NOT WIN32)
    set(GUI_MATC_ARGS --platform desktop)
endif()

set(GUI_MATERIAL_COMPILED_FILES "")
foreach(GUI_MAT_FILE ${GUI_MATERIAL_SOURCE_FILES})
    get_filename_component(GUI_MAT_NAME ${GUI_MAT_FILE} NAME_WE)
    set(GUI_MAT_SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${GUI_MAT_FILE}")
    set(GUI_MAT_OUTPUT_FILE "${GUI_RESOURCE_DIR}/${GUI_MAT_NAME}.filamat")

    add_custom_command(
        OUTPUT "${GUI_MAT_OUTPUT_FILE}"
        COMMAND ${FILAMENT_MATC} ${GUI_MATC_ARGS} -o "${GUI_MAT_OUTPUT_FILE}" "${GUI_MAT_SOURCE_FILE}"
		MAIN_DEPENDENCY ${GUI_MAT_FILE} DEPENDS ${FILAMENT_TARGET}
    )

    list(APPEND GUI_MATERIAL_COMPILED_FILES ${GUI_MAT_OUTPUT_FILE})
endforeach()

add_custom_target(materials ALL
    DEPENDS ${GUI_MATERIAL_COMPILED_FILES}
)

add_dependencies(GUI materials)

# Export GUI_RESOURCE_FILES to parent CMake context (cpp/open3d/)
set(GUI_RESOURCE_FILES
    ${GUI_RESOURCE_SOURCE_FILES} ${GUI_MATERIAL_COMPILED_FILES}
    PARENT_SCOPE)
set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)
