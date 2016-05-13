#version 120

varying vec2 UV;
uniform sampler2D image_texture;

void main()
{
	gl_FragColor = texture2D(image_texture, UV);
}