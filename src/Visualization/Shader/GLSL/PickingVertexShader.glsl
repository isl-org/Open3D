#version 330

in vec3 vertex_position;
in float vertex_index;
uniform mat4 MVP;

out vec4 fragment_color;

void main()
{
    float r, g, b, a;
    gl_Position = MVP * vec4(vertex_position, 1);
    r = floor(vertex_index / 16777216.0) / 255.0;
    g = mod(floor(vertex_index / 65536.0), 256.0) / 255.0;
    b = mod(floor(vertex_index / 256.0), 256.0) / 255.0;
    a = mod(vertex_index, 256.0) / 255.0;
    fragment_color = vec4(r, g, b, a);
}
