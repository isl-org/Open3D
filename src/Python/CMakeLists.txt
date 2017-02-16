file(GLOB PY_ALL_HEADER_FILES "*.h")
file(GLOB PY_ALL_SOURCE_FILES "*.cpp")
file(GLOB PY_CORE_HEADER_FILES "Core/*.h")
source_group("Header Files\\Core" FILES ${PY_CORE_HEADER_FILES})
file(GLOB PY_CORE_SOURCE_FILES "Core/*.cpp")
source_group("Source Files\\Core" FILES ${PY_CORE_SOURCE_FILES})
file(GLOB PY_IO_HEADER_FILES "IO/*.h")
source_group("Header Files\\IO" FILES ${PY_IO_HEADER_FILES})
file(GLOB PY_IO_SOURCE_FILES "IO/*.cpp")
source_group("Source Files\\IO" FILES ${PY_IO_SOURCE_FILES})
file(GLOB PY_VISUALIZATION_HEADER_FILES "Visualization/*.h")
source_group("Header Files\\Visualization" FILES ${PY_VISUALIZATION_HEADER_FILES})
file(GLOB PY_VISUALIZATION_SOURCE_FILES "Visualization/*.cpp")
source_group("Source Files\\Visualization" FILES ${PY_VISUALIZATION_SOURCE_FILES})
pybind11_add_module(py3d
	${PY_ALL_HEADER_FILES} ${PY_ALL_SOURCE_FILES}
  ${PY_CORE_SOURCE_FILES} ${PY_CORE_HEADER_FILES}
	${PY_IO_SOURCE_FILES} ${PY_IO_HEADER_FILES}
	${PY_VISUALIZATION_SOURCE_FILES} ${PY_VISUALIZATION_HEADER_FILES}
)
target_link_libraries(py3d PRIVATE Core IO Visualization)
