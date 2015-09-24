#version 120

varying vec3 vertex_position_world;
varying vec3 vertex_normal_camera;
varying vec3 eye_dir_camera;
varying vec3 light_dir_camera[3];
varying vec3 fragment_color;

uniform vec3 light_color[3];
uniform float light_power[3];

void main()
{
	vec3 diffuse_color = fragment_color;
	vec3 ambient_color = vec3(0.1, 0.1, 0.1) * diffuse_color;
	vec3 specular_color = vec3(0.3, 0.3, 0.3);
	float cos_theta[2];
	float cos_alpha[2];
	vec3 n, e, l, r;

	n = normalize(vertex_normal_camera);
	e = normalize(eye_dir_camera);
	l = normalize(light_dir_camera[0]);
	r = reflect(-l, n);
	cos_theta[0] = clamp(dot(n, l), 0, 1);
	cos_alpha[0] = clamp(dot(e, r), 0, 1);

	l= normalize(light_dir_camera[1]);
	r = reflect(-l, n);
	cos_theta[1] = clamp(dot(n, l), 0, 1);
	cos_alpha[1] = clamp(dot(e, r), 0, 1);

	l= normalize(light_dir_camera[2]);
	r = reflect(-l, n);
	cos_theta[2] = clamp(dot(n, l), 0, 1);
	cos_alpha[2] = clamp(dot(e, r), 0, 1);

	gl_FragColor.rgb = ambient_color + 
			diffuse_color * light_color[0] * light_power[0] * cos_theta[0] +
			specular_color * light_color[0] * light_power[0] * pow(cos_alpha[0], 5) +
			diffuse_color * light_color[1] * light_power[1] * cos_theta[1] +
			specular_color * light_color[1] * light_power[1] * pow(cos_alpha[1], 5) +
			diffuse_color * light_color[2] * light_power[2] * cos_theta[2] +
			specular_color * light_color[2] * light_power[2] * pow(cos_alpha[2], 5);
}
