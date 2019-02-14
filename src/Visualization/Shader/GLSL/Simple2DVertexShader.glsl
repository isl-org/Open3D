#version 330

in vec3 vertex_position;
in vec3 vertex_color;

out vec3 fragment_color;

void main()
{
    gl_Position = vec4(vertex_position, 1);
    fragment_color = vertex_color;
}
