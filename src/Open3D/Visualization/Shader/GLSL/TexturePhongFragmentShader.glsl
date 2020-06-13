#version 330

in vec3 vertex_position_world;
in vec3 vertex_normal_camera;
in vec3 eye_dir_camera;
in mat4 light_dir_camera_4;
in vec2 fragment_uv;

uniform mat4 light_color_4;
uniform vec4 light_diffuse_power_4;
uniform vec4 light_specular_power_4;
uniform vec4 light_specular_shininess_4;
uniform vec4 light_ambient;
uniform sampler2D diffuse_texture;

out vec4 FragColor;

void main()
{
    vec3 diffuse_color = texture(diffuse_texture, fragment_uv).rgb;
    vec3 ambient_color = light_ambient.xyz * diffuse_color;
    vec3 specular_color = vec3(1.0, 1.0, 1.0);
    vec4 cos_theta;
    vec4 cos_alpha;
    vec3 n, e, l, r;

    n = normalize(vertex_normal_camera);
    e = normalize(eye_dir_camera);
    l = normalize(light_dir_camera_4[0].xyz);
    r = reflect(-l, n);
    cos_theta[0] = clamp(dot(n, l), 0, 1);
    cos_alpha[0] = clamp(dot(e, r), 0, 1);

    l= normalize(light_dir_camera_4[1].xyz);
    r = reflect(-l, n);
    cos_theta[1] = clamp(dot(n, l), 0, 1);
    cos_alpha[1] = clamp(dot(e, r), 0, 1);

    l= normalize(light_dir_camera_4[2].xyz);
    r = reflect(-l, n);
    cos_theta[2] = clamp(dot(n, l), 0, 1);
    cos_alpha[2] = clamp(dot(e, r), 0, 1);

    l= normalize(light_dir_camera_4[3].xyz);
    r = reflect(-l, n);
    cos_theta[3] = clamp(dot(n, l), 0, 1);
    cos_alpha[3] = clamp(dot(e, r), 0, 1);

    FragColor = vec4(ambient_color + 
            diffuse_color * light_color_4[0].xyz * light_diffuse_power_4[0] * cos_theta[0] +
            specular_color * light_color_4[0].xyz * light_specular_power_4[0] * pow(cos_alpha[0], light_specular_shininess_4[0]) +
            diffuse_color * light_color_4[1].xyz * light_diffuse_power_4[1] * cos_theta[1] +
            specular_color * light_color_4[1].xyz * light_specular_power_4[1] * pow(cos_alpha[1], light_specular_shininess_4[1]) +
            diffuse_color * light_color_4[2].xyz * light_diffuse_power_4[2] * cos_theta[2] +
            specular_color * light_color_4[2].xyz * light_specular_power_4[2] * pow(cos_alpha[2], light_specular_shininess_4[2]) +
            diffuse_color * light_color_4[3].xyz * light_diffuse_power_4[3] * cos_theta[3] +
            specular_color * light_color_4[3].xyz * light_specular_power_4[3] * pow(cos_alpha[3], light_specular_shininess_4[3]), 1);
}
