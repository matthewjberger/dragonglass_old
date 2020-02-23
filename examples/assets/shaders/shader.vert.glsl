#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vCoords;

struct UBO {
  mat4 model;
  mat4 view;
  mat4 projection;
};

layout(binding = 0) uniform UniformBufferObjects {
  UBO data[100];
} ubos;

layout(push_constant) uniform Constants {
  vec4 baseColorFactor;
  int colorTextureSet;
  int uboIndex;
} constants;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragCoords;

void main() {

  fragColor = vColor;
  fragCoords = vCoords;

  gl_Position = ubos.data[constants.uboIndex].projection * ubos.data[constants.uboIndex].view * ubos.data[constants.uboIndex].model * vec4(vPosition, 1.0);

  // Flip the y coordinate when displaying gltf models
  // because Vulkan's coordinate system origin is in the top left
  // corner with the Y-axis pointing downwards
  // OpenGL's coordinate system origin is in the lower left with the
  // Y-axis pointing up
  gl_Position.y = -gl_Position.y;

}
