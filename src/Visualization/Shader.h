// Automatically generated header file for shader.
// See LICENSE.txt for full license statement.

#pragma once

namespace three {

namespace glsl {

const char * PointCloudFragmentShader = 
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

const char * PointCloudVertexShader = 
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

