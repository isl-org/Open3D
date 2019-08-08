#version 330

in vec2 UV;
uniform sampler2D depth_texture;

out vec4 FragColor;

float Interpolate(float value, float y0, float x0, float y1, float x1) {
    if (value < x0) return y0;
    if (value > x1) return y1;
    return (value - x0) * (y1 - y0) / (x1 - x0) + y0;
}

float Jet(float value /* already clamped in [0, 1] */) {
    if (value <= -0.75) {
        return 0.0;
    } else if (value <= -0.25) {
        return Interpolate(value, 0.0, -0.75, 1.0, -0.25);
    } else if (value <= 0.25) {
        return 1.0;
    } else if (value <= 0.75) {
        return Interpolate(value, 1.0, 0.25, 0.0, 0.75);
    } else {
        return 0.0;
    }
}

void main()
{
    float depth_min = 0.0;
    float depth_max = 3.0;
    float depth = texture(depth_texture, UV).r;
    depth = clamp(depth, depth_min, depth_max);
    depth = (depth - depth_min) / (depth_max - depth_min);
    FragColor = vec4(Jet(2 * depth - 1.5),
                     Jet(2 * depth- 1),
                     Jet(2 * depth - 0.5), 1);
}