#version 120

varying vec3 vertex_position_world;
varying vec3 vertex_normal_camera;
varying vec3 eye_dir_camera;
varying vec3 light_dir_camera;
varying vec3 fragment_color;

uniform vec3 light_position_world;
uniform vec3 light_color;
uniform float light_power;

void main()
{
	vec3 diffuse_color = fragment_color;
	vec3 ambient_color = vec3(0.1, 0.1, 0.1) * diffuse_color;
	vec3 specular_color = vec3(0.3, 0.3, 0.3);

	float distance = length(light_position_world - vertex_position_world);
	float distance2 = distance * distance;
	vec3 n = normalize(vertex_normal_camera);
	vec3 l = normalize(light_dir_camera);
	float cos_theta = clamp(dot(n, l), 0, 1);

	vec3 E = normalize(eye_dir_camera);
	vec3 R = reflect(-l, n);
	float cos_alpha = clamp(dot(E, R), 0, 1);

	gl_FragColor.rgb = ambient_color + 
			diffuse_color * light_color * light_power * cos_theta / distance2 +
			specular_color * light_color * light_power * pow(cos_alpha, 5) / distance2;
}
