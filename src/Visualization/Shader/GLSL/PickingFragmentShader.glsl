#version 400

varying vec4 fragment_color;
out vec4 FragColor;

void main()
{
    FragColor = fragment_color;
}
