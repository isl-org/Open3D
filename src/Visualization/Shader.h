// Automatically generated header file for shader.
// See LICENSE.txt for full license statement.

#pragma once

namespace three {

namespace glsl {

const char PointCloudVertexShader[] = 
"#version 120\n"
"\n"
"attribute vec3 vertexPosition_modelspace;\n"
"attribute vec3 vertexColor;\n"
"uniform mat4 MVP;\n"
"\n"
"varying vec3 fragmentColor;\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position =  MVP * vec4(vertexPosition_modelspace, 1);\n"
"	fragmentColor = vertexColor;\n"
"}\n"
"\n"
;

}  // namespace three::glsl

}  // namespace three

