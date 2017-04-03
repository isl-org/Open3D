
include_directories("${GLFW_SOURCE_DIR}/src"
                    "${GLFW_BINARY_DIR}/src"
                    ${glfw_INCLUDE_DIRS})

add_definitions(-D_GLFW_USE_CONFIG_H)

set(common_HEADERS internal.h
                   "${GLFW_BINARY_DIR}/src/glfw_config.h"
                   "${GLFW_SOURCE_DIR}/include/GLFW/glfw3.h"
                   "${GLFW_SOURCE_DIR}/include/GLFW/glfw3native.h")
set(common_SOURCES context.c init.c input.c monitor.c window.c)

if (_GLFW_COCOA)
    set(glfw_HEADERS ${common_HEADERS} cocoa_platform.h iokit_joystick.h
                     posix_tls.h)
    set(glfw_SOURCES ${common_SOURCES} cocoa_init.m cocoa_monitor.m
                     cocoa_window.m iokit_joystick.m mach_time.c posix_tls.c)
elseif (_GLFW_WIN32)
    set(glfw_HEADERS ${common_HEADERS} win32_platform.h win32_tls.h
                     winmm_joystick.h)
    set(glfw_SOURCES ${common_SOURCES} win32_init.c win32_monitor.c win32_time.c
                     win32_tls.c win32_window.c winmm_joystick.c)
elseif (_GLFW_X11)
    set(glfw_HEADERS ${common_HEADERS} x11_platform.h xkb_unicode.h
                     linux_joystick.h posix_time.h posix_tls.h)
    set(glfw_SOURCES ${common_SOURCES} x11_init.c x11_monitor.c x11_window.c
                     xkb_unicode.c linux_joystick.c posix_time.c posix_tls.c)
elseif (_GLFW_WAYLAND)
    set(glfw_HEADERS ${common_HEADERS} wl_platform.h linux_joystick.h
                     posix_time.h posix_tls.h xkb_unicode.h)
    set(glfw_SOURCES ${common_SOURCES} wl_init.c wl_monitor.c wl_window.c
                     linux_joystick.c posix_time.c posix_tls.c xkb_unicode.c)
elseif (_GLFW_MIR)
    set(glfw_HEADERS ${common_HEADERS} mir_platform.h linux_joystick.h
                     posix_time.h posix_tls.h xkb_unicode.h)
    set(glfw_SOURCES ${common_SOURCES} mir_init.c mir_monitor.c mir_window.c
                     linux_joystick.c posix_time.c posix_tls.c xkb_unicode.c)
endif()

if (_GLFW_EGL)
    list(APPEND glfw_HEADERS ${common_HEADERS} egl_context.h)
    list(APPEND glfw_SOURCES ${common_SOURCES} egl_context.c)
elseif (_GLFW_NSGL)
    list(APPEND glfw_HEADERS ${common_HEADERS} nsgl_context.h)
    list(APPEND glfw_SOURCES ${common_SOURCES} nsgl_context.m)
elseif (_GLFW_WGL)
    list(APPEND glfw_HEADERS ${common_HEADERS} wgl_context.h)
    list(APPEND glfw_SOURCES ${common_SOURCES} wgl_context.c)
elseif (_GLFW_X11)
    list(APPEND glfw_HEADERS ${common_HEADERS} glx_context.h)
    list(APPEND glfw_SOURCES ${common_SOURCES} glx_context.c)
endif()

if (APPLE)
    # For some reason, CMake doesn't know about .m
	set_source_files_properties(${glfw_SOURCES} PROPERTIES LANGUAGE C)
	# Suppress a warning created by this version of glfw
    add_compile_options(-Wno-deprecated)
endif()

add_library(glfw ${glfw_SOURCES} ${glfw_HEADERS})
target_link_libraries(glfw ${glfw_LIBRARIES})
set_target_properties(glfw PROPERTIES
                      OUTPUT_NAME "${GLFW_LIB_NAME}"
                      VERSION ${GLFW_VERSION}
                      SOVERSION ${GLFW_VERSION_MAJOR}
                      POSITION_INDEPENDENT_CODE ON
                      FOLDER "External")

install(TARGETS glfw EXPORT glfwTargets DESTINATION lib)
