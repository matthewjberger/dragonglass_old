#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 vPosition;

layout (binding = 0) uniform Ubo {
  mat4 model;
  mat4 view;
  mat4 projection;
} ubo;

void main() {
  gl_Position = ubo.projection * ubo.view * ubo.model * vec4(vPosition, 1.0);
  gl_Position.y = -gl_Position.y;
}
