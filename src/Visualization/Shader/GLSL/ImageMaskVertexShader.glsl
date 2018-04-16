#version 120

attribute vec3 vertex_position;
attribute vec2 vertex_UV;

varying vec2 UV;

void main()
{
	gl_Position = vec4(vertex_position, 1);
	UV = vertex_UV;
}
