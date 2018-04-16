#version 120

varying vec2 UV;
uniform sampler2D image_texture;

uniform vec3 mask_color;
uniform float mask_alpha;

void main()
{
	gl_FragColor = vec4(mask_color, texture2D(image_texture, UV).r * mask_alpha);
}
