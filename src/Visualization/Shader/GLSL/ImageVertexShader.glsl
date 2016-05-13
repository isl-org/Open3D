#version 120

attribute vec3 vertex_position;
attribute vec2 vertex_UV;

varying vec2 UV;

uniform vec3 vertex_scale;

void main()
{
	gl_Position = vec4(vertex_position * vertex_scale, 1);
	UV = vertex_UV;
}
