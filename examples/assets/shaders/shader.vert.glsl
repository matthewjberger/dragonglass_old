#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vCoords;

layout(binding = 0) uniform UboView {
  mat4 view;
  mat4 projection;
  vec3 cameraposition;
} uboView;

layout(binding = 1) uniform UboInstance {
  mat4 model;
} uboInstance;

layout(push_constant) uniform Constants {
  vec4 baseColorFactor;
  int colorTextureSet;
} constants;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragCoords;
layout(location = 2) out vec3 fragPosition;
layout(location = 3) out vec3 fragCameraPosition;

void main() {
  vec4 position = uboInstance.model * vec4(vPosition, 1.0);
  position.y = -position.y;

  fragNormal = mat3(transpose(inverse(uboInstance.model))) * vNormal;
  fragCoords = vCoords;
  fragPosition = position.xyz;
  fragCameraPosition = uboView.cameraposition;

  gl_Position = uboView.projection * uboView.view * position;
}
