#version 120

attribute vec3 vertex_position;
attribute vec3 vertex_color;
uniform mat4 MVP;

varying vec3 fragment_color;

void main()
{
	gl_Position = MVP * vec4(vertex_position, 1);
	fragment_color = vertex_color;
}
