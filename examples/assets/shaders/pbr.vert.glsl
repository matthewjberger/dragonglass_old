#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec2 inUV1;
layout (location = 4) in vec4 inJoint0;
layout (location = 5) in vec4 inWeight0;

layout(binding = 0) uniform UboView {
  mat4 view;
  mat4 projection;
  vec4 cameraPosition;
} uboView;

#define MAX_NUM_JOINTS 20

layout(binding = 1) uniform UboInstance {
  mat4 model;
  mat4 jointMatrices[MAX_NUM_JOINTS];
  float jointCount;
} uboInstance;

layout(push_constant) uniform Constants {
  vec4 baseColorFactor;
  int colorTextureSet;
} constants;

layout (location = 0) out vec3 outWorldPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV0;
layout (location = 3) out vec2 outUV1;

void main()
{
  mat4 skinMatrix = mat4(1.0);
  if (uboInstance.jointCount > 0.0) {
    skinMatrix =
      inWeight0.x * uboInstance.jointMatrices[int(inJoint0.x)] +
      inWeight0.y * uboInstance.jointMatrices[int(inJoint0.y)] +
      inWeight0.z * uboInstance.jointMatrices[int(inJoint0.z)] +
      inWeight0.w * uboInstance.jointMatrices[int(inJoint0.w)];
  }
  vec4 locPos;
  locPos = uboInstance.model * skinMatrix * vec4(inPos, 1.0);
  outNormal = normalize(transpose(inverse(mat3(uboInstance.model * skinMatrix))) * inNormal);
  locPos.y = -locPos.y;
  outWorldPos = locPos.xyz / locPos.w;
  outUV0 = inUV0;
  outUV1 = inUV1;
  gl_Position =  uboView.projection * uboView.view * vec4(outWorldPos, 1.0);
}
