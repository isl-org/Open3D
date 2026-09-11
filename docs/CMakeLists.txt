configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/getting_started.in.rst
    ${CMAKE_CURRENT_SOURCE_DIR}/getting_started.rst @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dev_wheels.in.rst
    ${CMAKE_CURRENT_SOURCE_DIR}/dev_wheels.rst @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sycl.in.rst
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl.rst @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docker.in.rst
    ${CMAKE_CURRENT_SOURCE_DIR}/docker.rst @ONLY)
