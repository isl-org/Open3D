#version 330

in vec2 UV;
uniform sampler2D image_texture;


/* built-in option to ensure RGB and D are handled in the same shader,
   which can be used in 2 passes */
#define DEPTH_TEXTURE_MODE 0
#define RGB_TEXTURE_MODE 1
#define GRAYSCALE_TEXTURE_MODE 2
uniform int texture_mode;

/* Decides the colormap of the depth image */
uniform float depth_max;
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

void main() {
    if (texture_mode == DEPTH_TEXTURE_MODE) {
        float depth = texture(image_texture, UV).r;
        depth = clamp(depth, 0, depth_max);
        depth = depth / depth_max;
        depth = 2 * depth - 1;
        FragColor = vec4(Jet(depth - 0.5), Jet(depth), Jet(depth + 0.5), 1);
    } else if (texture_mode == RGB_TEXTURE_MODE) {
        FragColor = texture(image_texture, UV);
    } else if (texture_mode == GRAYSCALE_TEXTURE_MODE) {
        float scalar = texture(image_texture, UV).r;
        FragColor = vec4(vec3(scalar), 1);
    }
}