#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 vPosition;

layout(binding = 0) uniform Ubo {
  mat4 view;
  mat4 projection;
} ubo;

layout(location = 0) out vec3 vert_texcoord;

void main() {
  gl_Position = ubo.projection * mat4(mat3(ubo.view)) * vec4(vPosition.xyz, 1.0);
  vert_texcoord = vPosition;
}
