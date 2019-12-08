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
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(vPosition, 1.0);
    fragColor = vColor;
    fragCoords = vCoords;
}
