// Automatically generated header file for shader.
// See LICENSE.txt for full license statement.

#pragma once

namespace three {

namespace glsl {

const char * const PointCloudFragmentShader = 
"#version 120\n"
"\n"
"varying vec3 fragment_color;\n"
"\n"
"void main()\n"
"{\n"
"	gl_FragColor = vec4(fragment_color, 1);\n"
"}\n"
;

}  // namespace three::glsl

}  // namespace three

namespace three {

namespace glsl {

const char * const PointCloudVertexShader = 
"#version 120\n"
"\n"
"attribute vec3 vertex_position;\n"
"attribute vec3 vertex_color;\n"
"uniform mat4 MVP;\n"
"\n"
"varying vec3 fragment_color;\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position =  MVP * vec4(vertex_position, 1);\n"
"	fragment_color = vertex_color;\n"
"}\n"
;

}  // namespace three::glsl

}  // namespace three

namespace three {

namespace glsl {

const char * const TriangleMeshFragmentShader = 
"#version 120\n"
"\n"
"varying vec3 vertex_position_world;\n"
"varying vec3 vertex_normal_camera;\n"
"varying vec3 eye_dir_camera;\n"
"varying vec3 light_dir_camera[3];\n"
"varying vec3 fragment_color;\n"
"\n"
"uniform vec3 light_color[3];\n"
"uniform float light_power[3];\n"
"\n"
"void main()\n"
"{\n"
"	vec3 diffuse_color = fragment_color;\n"
"	vec3 ambient_color = vec3(0.1, 0.1, 0.1) * diffuse_color;\n"
"	vec3 specular_color = vec3(0.3, 0.3, 0.3);\n"
"	float cos_theta[2];\n"
"	float cos_alpha[2];\n"
"	vec3 n, e, l, r;\n"
"\n"
"	n = normalize(vertex_normal_camera);\n"
"	e = normalize(eye_dir_camera);\n"
"	l = normalize(light_dir_camera[0]);\n"
"	r = reflect(-l, n);\n"
"	cos_theta[0] = clamp(dot(n, l), 0, 1);\n"
"	cos_alpha[0] = clamp(dot(e, r), 0, 1);\n"
"\n"
"	l= normalize(light_dir_camera[1]);\n"
"	r = reflect(-l, n);\n"
"	cos_theta[1] = clamp(dot(n, l), 0, 1);\n"
"	cos_alpha[1] = clamp(dot(e, r), 0, 1);\n"
"\n"
"	l= normalize(light_dir_camera[2]);\n"
"	r = reflect(-l, n);\n"
"	cos_theta[2] = clamp(dot(n, l), 0, 1);\n"
"	cos_alpha[2] = clamp(dot(e, r), 0, 1);\n"
"\n"
"	gl_FragColor.rgb = ambient_color + \n"
"			diffuse_color * light_color[0] * light_power[0] * cos_theta[0] +\n"
"			specular_color * light_color[0] * light_power[0] * pow(cos_alpha[0], 5) +\n"
"			diffuse_color * light_color[1] * light_power[1] * cos_theta[1] +\n"
"			specular_color * light_color[1] * light_power[1] * pow(cos_alpha[1], 5) +\n"
"			diffuse_color * light_color[2] * light_power[2] * cos_theta[2] +\n"
"			specular_color * light_color[2] * light_power[2] * pow(cos_alpha[2], 5);\n"
"}\n"
;

}  // namespace three::glsl

}  // namespace three

namespace three {

namespace glsl {

const char * const TriangleMeshVertexShader = 
"#version 120\n"
"\n"
"attribute vec3 vertex_position;\n"
"attribute vec3 vertex_normal;\n"
"attribute vec3 vertex_color;\n"
"\n"
"varying vec3 vertex_position_world;\n"
"varying vec3 vertex_normal_camera;\n"
"varying vec3 eye_dir_camera;\n"
"varying vec3 light_dir_camera[3];\n"
"varying vec3 fragment_color;\n"
"\n"
"uniform mat4 MVP;\n"
"uniform mat4 V;\n"
"uniform mat4 M;\n"
"uniform vec3 light_position_world[3];\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position =  MVP * vec4(vertex_position, 1);\n"
"	vertex_position_world = (M * vec4(vertex_position, 1)).xyz;\n"
"\n"
"	vec3 vertex_position_camera = (V * M * vec4(vertex_position, 1)).xyz;\n"
"	eye_dir_camera = vec3(0, 0, 0) - vertex_position_camera;\n"
"\n"
"	vec3 light_position_camera = (V * vec4(light_position_world[0], 1)).xyz;\n"
"	light_dir_camera[0] = light_position_camera - vertex_position_camera;\n"
"	light_position_camera = (V * vec4(light_position_world[1], 1)).xyz;\n"
"	light_dir_camera[1] = light_position_camera - vertex_position_camera;\n"
"	light_position_camera = (V * vec4(light_position_world[2], 1)).xyz;\n"
"	light_dir_camera[2] = light_position_camera - vertex_position_camera;\n"
"\n"
"	vertex_normal_camera = (V * M * vec4(vertex_normal, 0)).xyz;\n"
"\n"
"	fragment_color = vertex_color;\n"
"}\n"
;

}  // namespace three::glsl

}  // namespace three

