#version 330

in vec3 vertex_normal_camera;
out vec4 FragColor;

void main()
{
    FragColor = vec4(vertex_normal_camera * 0.5 + 0.5, 1);
}
