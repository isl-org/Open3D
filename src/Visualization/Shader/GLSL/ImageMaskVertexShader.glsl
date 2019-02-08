#version 400

in vec3 vertex_position;
in vec2 vertex_UV;

varying vec2 UV;

void main()
{
    gl_Position = vec4(vertex_position, 1);
    UV = vertex_UV;
}
