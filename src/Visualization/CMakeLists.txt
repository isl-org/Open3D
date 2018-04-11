# some black magic to automatically create a header file from shader files
file(GLOB SHADER_FILES "Shader/GLSL/*.glsl")
add_custom_command(
	OUTPUT ${Open3D_SOURCE_DIR}/Visualization/Shader/Shader.h
	COMMAND EncodeShader Shader.h
	WORKING_DIRECTORY ${Open3D_SOURCE_DIR}/Visualization/Shader/
	DEPENDS ${SHADER_FILES} EncodeShader
)
foreach(file ${SHADER_FILES})
	add_custom_command(
		OUTPUT ${Open3D_SOURCE_DIR}/Visualization/Shader/Shader.h
		COMMAND EncodeShader Shader.h ${file}
		WORKING_DIRECTORY ${Open3D_SOURCE_DIR}/Visualization/Shader/
		DEPENDS ${SHADER_FILES} EncodeShader
		APPEND
	)
endforeach(file)
add_custom_target(ShaderFileTarget ALL
DEPENDS ${Open3D_SOURCE_DIR}/Visualization/Shader/Shader.h)
set_target_properties(ShaderFileTarget PROPERTIES FOLDER "Custom")

source_group("Source Files\\Shader\\GLSL" FILES ${SHADER_FILES})
file(GLOB VISUALIZATION_ALL_HEADER_FILES "*.h")
file(GLOB VISUALIZATION_ALL_SOURCE_FILES "*.cpp")
file(GLOB VISUALIZATION_UTILITY_HEADER_FILES "Utility/*.h")
source_group("Header Files\\Utility" FILES ${VISUALIZATION_UTILITY_HEADER_FILES})
file(GLOB VISUALIZATION_UTILITY_SOURCE_FILES "Utility/*.cpp")
source_group("Source Files\\Utility" FILES ${VISUALIZATION_UTILITY_SOURCE_FILES})
file(GLOB VISUALIZATION_SHADER_HEADER_FILES "Shader/*.h")
source_group("Header Files\\Shader" FILES ${VISUALIZATION_SHADER_HEADER_FILES})
file(GLOB VISUALIZATION_SHADER_SOURCE_FILES "Shader/*.cpp")
source_group("Source Files\\Shader" FILES ${VISUALIZATION_SHADER_SOURCE_FILES})
file(GLOB VISUALIZATION_VISUALIZER_HEADER_FILES "Visualizer/*.h")
source_group("Header Files\\Visualizer" FILES ${VISUALIZATION_VISUALIZER_HEADER_FILES})
file(GLOB VISUALIZATION_VISUALIZER_SOURCE_FILES "Visualizer/*.cpp")
source_group("Source Files\\Visualizer" FILES ${VISUALIZATION_VISUALIZER_SOURCE_FILES})

add_library(Visualization ${Open3D_LIB_TYPE}
	${SHADER_FILES}
	${VISUALIZATION_ALL_HEADER_FILES} ${VISUALIZATION_ALL_SOURCE_FILES}
	${VISUALIZATION_UTILITY_HEADER_FILES} ${VISUALIZATION_UTILITY_SOURCE_FILES}
	${VISUALIZATION_SHADER_HEADER_FILES} ${VISUALIZATION_SHADER_SOURCE_FILES}
	${VISUALIZATION_VISUALIZER_HEADER_FILES} ${VISUALIZATION_VISUALIZER_SOURCE_FILES}
)
add_dependencies(Visualization ShaderFileTarget)

if (Open3D_HEADLESS_RENDERING)
	find_package(OSMesa REQUIRED)
	include_directories(${OSMESA_INCLUDE_DIR})
	target_link_libraries(Visualization Core ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OSMESA_LIBRARY})
else ()
	find_package(OpenGL REQUIRED)
	target_link_libraries(Visualization Core ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
endif ()


install(TARGETS Visualization
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/Open3D
)

install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/GeometryRenderer.h"                   DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/ImageMaskShader.h"                    DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/ImageShader.h"                        DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/NormalShader.h"                       DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/PhongShader.h"                        DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/PickingShader.h"                      DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/Shader.h"                             DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/ShaderWrapper.h"                      DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/Simple2DShader.h"                     DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/SimpleBlackShader.h"                  DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Shader/SimpleShader.h"                       DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Shader)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Utility/BoundingBox.h"                       DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Utility)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Utility/ColorMap.h"                          DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Utility)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Utility/DrawGeometry.h"                      DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Utility)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Utility/GLHelper.h"                          DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Utility)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Utility/PointCloudPicker.h"                  DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Utility)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Utility/SelectionPolygon.h"                  DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Utility)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Utility/SelectionPolygonVolume.h"            DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Utility)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualization.h"                             DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/RenderOption.h"                   DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/RenderOptionWithEditing.h"        DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/ViewControl.h"                    DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/ViewControlWithCustomAnimation.h" DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/ViewControlWithEditing.h"         DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/ViewParameters.h"                 DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/ViewTrajectory.h"                 DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/Visualizer.h"                     DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/VisualizerWithCustomAnimation.h"  DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/VisualizerWithEditing.h"          DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
install(FILES "${CMAKE_SOURCE_DIR}/Visualization/Visualizer/VisualizerWithKeyCallback.h"      DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/Visualization/Visualizer)
