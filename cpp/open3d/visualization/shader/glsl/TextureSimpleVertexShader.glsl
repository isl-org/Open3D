#version 330

in vec3 vertex_position;
in vec2 vertex_uv;
uniform mat4 MVP;

out vec2 fragment_uv;

void main()
{
    gl_Position = MVP * vec4(vertex_position, 1);
    fragment_uv = vertex_uv;
}
