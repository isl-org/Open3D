macro(APP SRC_DIR APP_NAME TARGET_NAME)
    set(APPS_DIR "${PROJECT_SOURCE_DIR}/cpp/apps")
    set(SOURCE_DIR "${APPS_DIR}/${SRC_DIR}")

    file(GLOB SOURCE_FILES "${SOURCE_DIR}/*.cpp")
    file(GLOB HEADER_FILES "${SOURCE_DIR}/*.h")

    if (APPLE)
        file(GLOB OBJC_FILES "${SOURCE_DIR}/*.mm")
        set(SOURCE_FILES ${SOURCE_FILES} ${OBJC_FILES})
        set(ICON_FILE "${SOURCE_DIR}/AppIcon.icns")

        file(GLOB RESOURCE_FILES "${SOURCE_DIR}/*.icns")
        list(APPEND RESOURCE_FILES "${SOURCE_DIR}/Assets.car")

        set(INFO_PLIST "${SOURCE_DIR}/Info.plist.in")
        
        set(MACOSX_BUNDLE_NAME ${APP_NAME})
        set(MACOSX_BUNDLE_EXECUTABLE_NAME ${APP_NAME})
        set(MACOSX_BUNDLE_GUI_IDENTIFIER com.intel-isl.open3d.${APP_NAME})
        set(MACOSX_BUNDLE_ICON_FILE "AppIcon")
        set(MACOSX_BUNDLE_LONG_VERSION_STRING ${PROJECT_VERSION_THREE_NUMBER})
        set(MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_THREE_NUMBER})
        set(MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION_THREE_NUMBER})
        set(MACOSX_BUNDLE_COPYRIGHT "Copyright (C) 2020 by Open3D")
        add_executable(${TARGET_NAME} ${SOURCE_FILES} ${HEADER_FILES})
        set_target_properties(${TARGET_NAME} PROPERTIES
                              MACOSX_BUNDLE TRUE
                              MACOSX_BUNDLE_INFO_PLIST "${INFO_PLIST}"
                              XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "" # disable
                              OUTPUT_NAME ${APP_NAME}
                             )
    else()
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${APP_NAME}")
        add_executable(${TARGET_NAME} ${SOURCE_FILES} ${HEADER_FILES})
        set_target_properties(${TARGET_NAME} PROPERTIES
                              OUTPUT_NAME ${APP_NAME}
                             )
    endif()
    add_dependencies(${TARGET_NAME} GuiResources)

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${TARGET_NAME} PRIVATE ${DEPENDENCY})
    endforeach()

    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "apps")
    open3d_show_and_abort_on_warning(${TARGET_NAME})
    open3d_set_global_properties(${TARGET_NAME})

    # Copy the resource files. This needs to be done as a post-build step
    # because on macOS, we don't know the bundle directory until build time
    # if we are using Xcode.
    set(RESOURCE_FILES ${GUI_RESOURCE_FILES} ${RESOURCE_FILES})
    if (APPLE)
        set(RESOURCE_DIR_NAME "Contents/Resources")
    else ()
        set(RESOURCE_DIR_NAME "resources")
    endif()

    # $<TARGET_BUNDLE_DIR> does not exist at config time, so we need to
    # duplicate the post build step on macOS and non-macOS
    if (APPLE)
        add_custom_command(TARGET "${TARGET_NAME}"
            POST_BUILD
            # copy the resource files into the bundle
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>/${RESOURCE_DIR_NAME}"
            COMMAND ${CMAKE_COMMAND} -E copy ${RESOURCE_FILES} "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>/${RESOURCE_DIR_NAME}"
            # copy external libraries (e.g. SDL into the bundle and fixup
            # the search paths
            COMMAND ${APPS_DIR}/fixup_macosx_bundle.sh "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>"
        )
    else ()
        set(APP_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        add_custom_command(TARGET "${TARGET_NAME}"
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${APP_DIR}/${RESOURCE_DIR_NAME}"
            COMMAND ${CMAKE_COMMAND} -E copy ${RESOURCE_FILES} "${APP_DIR}/${RESOURCE_DIR_NAME}"
        )
        if (UNIX)
            install(DIRECTORY   "${APP_DIR}"
                    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
                    USE_SOURCE_PERMISSIONS)
            if (CMAKE_INSTALL_PREFIX MATCHES "^(/usr/local|/opt)")
                set(DESKTOP_INSTALL_DIR "/usr/share")
            else()
                set(DESKTOP_INSTALL_DIR "$ENV{HOME}/.local/share")
            endif()
            configure_file("${SOURCE_DIR}/${TARGET_NAME}.desktop.in"
                           "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${APP_NAME}.desktop")
            # Install using freedesktop.org standards
            install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${APP_NAME}.desktop"
                    DESTINATION "${DESKTOP_INSTALL_DIR}/applications")
            install(FILES "${SOURCE_DIR}/icon.svg"
                    DESTINATION "${DESKTOP_INSTALL_DIR}/icons/hicolor/scalable/apps"
                    RENAME "${APP_NAME}.svg")
            install(FILES "${SOURCE_DIR}/${TARGET_NAME}.xml"
                    DESTINATION "${DESKTOP_INSTALL_DIR}/mime/packages"
                    RENAME "${APP_NAME}.xml")
            # Various caches need to be updated for the app to become visible
            install(CODE "execute_process(COMMAND ${SOURCE_DIR}/postinstall-linux.sh)")
        endif()
    endif()
endmacro(APP)

if (ENABLE_GUI)
    APP(Open3DViewer Open3D Open3DViewer ${CMAKE_PROJECT_NAME})
endif()
