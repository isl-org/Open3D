#version 120

attribute vec3 vertex_position;
attribute vec3 vertex_normal;
attribute vec3 vertex_color;

varying vec3 vertex_position_world;
varying vec3 vertex_normal_camera;
varying vec3 eye_dir_camera;
varying vec3 fragment_color;
varying mat4 light_dir_camera_4;

uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;
uniform mat4 light_position_world_4;

void main()
{
	gl_Position = MVP * vec4(vertex_position, 1);
	vertex_position_world = (M * vec4(vertex_position, 1)).xyz;

	vec3 vertex_position_camera = (V * M * vec4(vertex_position, 1)).xyz;
	eye_dir_camera = vec3(0, 0, 0) - vertex_position_camera;

	vec4 v = vec4(vertex_position_camera, 1);
	light_dir_camera_4 = V * light_position_world_4 - mat4(v, v, v, v);

	vertex_normal_camera = (V * M * vec4(vertex_normal, 0)).xyz;
	if (dot(eye_dir_camera, vertex_normal_camera) < 0.0)
		vertex_normal_camera = vertex_normal_camera * -1.0;

	fragment_color = vertex_color;
}
