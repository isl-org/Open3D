configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/getting_started.in.rst
    ${CMAKE_CURRENT_SOURCE_DIR}/getting_started.rst @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docker.in.rst
    ${CMAKE_CURRENT_SOURCE_DIR}/docker.rst @ONLY)

include(FetchContent)

FetchContent_Declare(
    ext_open3d_sphinx_theme
    PREFIX open3d_sphinx_theme
    URL https://github.com/isl-org/open3d_sphinx_theme/archive/c71d2728eb5afd1aeeb20dc27a5a0d42bb402d83.tar.gz
    URL_HASH SHA256=98af8b7fdb75a74280b6187dbb58ea601db978d4f3f8956d3d87c59c20786f73
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/open3d_sphinx_theme"
)

if(NOT ext_open3d_sphinx_theme_POPULATED)
  FetchContent_Populate(ext_open3d_sphinx_theme)

  set(OPEN3D_SPHINX_THEME_SOURCE_DIR ${ext_open3d_sphinx_theme_SOURCE_DIR})
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.in.py
    ${CMAKE_CURRENT_SOURCE_DIR}/conf.py @ONLY)
