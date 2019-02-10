#version 330 core

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec2 vertex_UV;

out vec2 UV;

uniform vec3 vertex_scale;

void main()
{
    gl_Position = vec4(vertex_position * vertex_scale, 1);
    UV = vertex_UV;
}
