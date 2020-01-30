macro(APP APP_NAME)
    set(APPS_DIR "${PROJECT_SOURCE_DIR}/src/Apps")
    set(SOURCE_DIR "${APPS_DIR}/${APP_NAME}")

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
        set(MACOSX_BUNDLE_GUI_IDENTIFIER com.intel-isl.open3d.${APP_NAME})
        set(MACOSX_BUNDLE_ICON_FILE "AppIcon")
        set(MACOSX_BUNDLE_LONG_VERSION_STRING ${PROJECT_VERSION_THREE_NUMBER})
        set(MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_THREE_NUMBER})
        set(MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION_THREE_NUMBER})
        set(MACOSX_BUNDLE_COPYRIGHT "Copyright (C) 2020 by Open3D")
        add_executable(${APP_NAME} ${SOURCE_FILES} ${HEADER_FILES})
        set_target_properties(${APP_NAME} PROPERTIES
                              MACOSX_BUNDLE TRUE
                              MACOSX_BUNDLE_INFO_PLIST "${INFO_PLIST}"
                              XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "" # disable
                             )
    else()
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${APP_NAME}")
        add_executable(${APP_NAME} ${SOURCE_FILES} ${HEADER_FILES})
    endif()
    add_dependencies(${APP_NAME} GuiResources)

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${APP_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${APP_NAME} PROPERTIES FOLDER "Apps")
    ShowAndAbortOnWarning(${APP_NAME})

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
        add_custom_command(TARGET "${APP_NAME}"
            POST_BUILD
            # copy the resource files into the bundle
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_DIR:${APP_NAME}>/${RESOURCE_DIR_NAME}"
            COMMAND ${CMAKE_COMMAND} -E copy ${RESOURCE_FILES} "$<TARGET_BUNDLE_DIR:${APP_NAME}>/${RESOURCE_DIR_NAME}"
            # copy external libraries (e.g. SDL into the bundle and fixup
            # the search paths
            COMMAND ${APPS_DIR}/fixup_macosx_bundle.sh "$<TARGET_BUNDLE_DIR:${APP_NAME}>"
        )
    else ()
        set(APP_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        add_custom_command(TARGET "${APP_NAME}"
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${APP_DIR}/${RESOURCE_DIR_NAME}"
            COMMAND ${CMAKE_COMMAND} -E copy ${RESOURCE_FILES} "${APP_DIR}/${RESOURCE_DIR_NAME}"
        )
    endif()
endmacro(APP)

APP(Open3DViewer ${CMAKE_PROJECT_NAME} SDL2)
