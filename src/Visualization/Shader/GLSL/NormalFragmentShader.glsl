#version 400

varying vec3 vertex_normal_camera;
out vec4 fragmentColor;

void main()
{
    fragmentColor = vertex_normal_camera * 0.5 + 0.5;
}
