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
  vec3 position = mat3(ubo.view) * vPosition;
  gl_Position = (ubo.projection * vec4(position, 0.0)).xyzz;
  vert_texcoord = vPosition;
  vert_texcoord.y *= -1.0;
}
