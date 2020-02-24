#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragCoords;

layout(binding = 2) uniform sampler2D textures[100];

layout(push_constant) uniform Material {
  vec4 baseColorFactor;
  int colorTextureSet;
} material;

layout(location = 0) out vec4 outColor;

void main() {
  if (material.colorTextureSet > -1)
  {
    outColor = texture(textures[material.colorTextureSet], fragCoords);
  } else {
    outColor = material.baseColorFactor;
  }
}
