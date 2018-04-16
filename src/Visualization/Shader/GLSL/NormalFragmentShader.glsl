#version 120

varying vec3 vertex_normal_camera;

void main()
{
	gl_FragColor.rgb = vertex_normal_camera * 0.5 + 0.5;
}
