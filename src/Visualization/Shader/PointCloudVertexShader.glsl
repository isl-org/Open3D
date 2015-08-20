#version 120

attribute vec3 vertexPosition_modelspace;
attribute vec3 vertexColor;
uniform mat4 MVP;

varying vec3 fragmentColor;

void main()
{
	gl_Position =  MVP * vec4(vertexPosition_modelspace, 1);
	fragmentColor = vertexColor;
}

