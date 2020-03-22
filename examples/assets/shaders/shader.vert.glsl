#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vCoords;

layout(binding = 0) uniform UboView {
  mat4 view;
  mat4 projection;
} uboView;

layout(binding = 1) uniform UboInstance {
  mat4 model;
} uboInstance;

layout(push_constant) uniform Constants {
  vec4 baseColorFactor;
  int colorTextureSet;
} constants;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragCoords;

void main() {

  fragColor = vColor;
  fragCoords = vCoords;

  // Flip the y coordinate when displaying gltf models
  // because Vulkan's coordinate system origin is in the top left
  // corner with the Y-axis pointing downwards
  // OpenGL's coordinate system origin is in the lower left with the Y-axis pointing up
  vec3 position = vPosition;
  position.y *= -1.0;

  gl_Position = uboView.projection * uboView.view * uboInstance.model * vec4(position, 1.0);
}
