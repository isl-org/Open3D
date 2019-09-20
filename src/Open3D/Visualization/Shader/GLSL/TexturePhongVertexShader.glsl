#version 330

in vec3 vertex_position;
in vec3 vertex_normal;
in vec2 vertex_uv;

out vec3 vertex_position_world;
out vec3 vertex_normal_camera;
out vec3 eye_dir_camera;
out mat4 light_dir_camera_4;
out vec2 fragment_uv;

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

  fragment_uv = vertex_uv;
}

