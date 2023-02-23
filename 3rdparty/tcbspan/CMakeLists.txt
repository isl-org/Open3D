cmake_minimum_required(VERSION 3.8)
project(span LANGUAGES CXX)

option(BUILD_RAYCAST_APP "If the raycast test app should be built" OFF)

include(GNUInstallDirs)

add_library(span INTERFACE)
add_library(tcb::span ALIAS span)
target_include_directories(span INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/tcb/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)

install(TARGETS span EXPORT tcbspan-targets)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/tcb/span.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tcbspan/
)

install(EXPORT tcbspan-targets
    FILE tcbspan-targets.cmake
    NAMESPACE tcb::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tcbspan/
)

install(FILES LICENSE_1_0.txt
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/tcbspan/
)