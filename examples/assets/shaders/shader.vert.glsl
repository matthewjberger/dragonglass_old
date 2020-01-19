#version 450

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vCoords;

layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 projection;
} ubo;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragCoords;

void main() {

  fragColor = vColor;
  fragCoords = vCoords;

  gl_Position = ubo.projection * ubo.view * ubo.model * vec4(vPosition, 1.0);

  // Flip the y coordinate when displaying gltf models
  // because Vulkan's coordinate system origin is in the top left
  // corner with the Y-axis pointing downwards
  // OpenGL's coordinate system origin is in the lower left with the
  // Y-axis pointing up
  gl_Position.y = -gl_Position.y;

}