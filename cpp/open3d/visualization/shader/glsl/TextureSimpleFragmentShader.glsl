#version 330

in vec2 fragment_uv;
out vec4 FragColor;

uniform sampler2D diffuse_texture;

void main()
{
    FragColor = texture(diffuse_texture, fragment_uv);
}
