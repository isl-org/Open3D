#version 330

in vec2 UV;
uniform sampler2D image_texture;

uniform vec3 mask_color;
uniform float mask_alpha;

out vec4 FragColor;

void main()
{
    FragColor = vec4(mask_color, texture(image_texture, UV).r * mask_alpha);
}
