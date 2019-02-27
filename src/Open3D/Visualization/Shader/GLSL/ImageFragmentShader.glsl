#version 330

in vec2 UV;
uniform sampler2D image_texture;

out vec4 FragColor;

void main()
{
    FragColor = texture(image_texture, UV);
}