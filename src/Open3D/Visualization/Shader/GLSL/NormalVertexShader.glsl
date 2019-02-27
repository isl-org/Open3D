#version 330

in vec3 vertex_position;
in vec3 vertex_normal;

out vec3 vertex_normal_camera;

uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;

void main()
{
    gl_Position = MVP * vec4(vertex_position, 1);
    vertex_normal_camera = (V * M * vec4(vertex_normal, 0)).xyz;
}
