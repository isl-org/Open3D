#version 120

attribute vec3 vertex_position;
attribute vec3 vertex_color;

varying vec3 fragment_color;

void main()
{
	gl_Position = vec4(vertex_position, 1);
	fragment_color = vertex_color;
}
