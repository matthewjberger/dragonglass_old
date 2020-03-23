#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragCoords;
layout(location = 2) in vec3 fragPosition;
layout(location = 3) in vec3 fragCameraPosition;

layout(binding = 2) uniform sampler2D textures[100];
layout(binding = 3) uniform samplerCube cubemap;

layout(push_constant) uniform Material {
  vec4 baseColorFactor;
  int colorTextureSet;
} material;

layout(location = 0) out vec4 outColor;

void main() {
  // mirror
  // vec3 I = normalize(fragPosition - fragCameraPosition);
  // vec3 R = reflect(I, normalize(fragNormal));
  // outColor = vec4(texture(cubemap, R).rgb, 1.0);

  // glass
  // vec3 I = normalize(fragPosition - fragCameraPosition);
  // float refractive_index = 1.00 / 1.52;
  // vec3 R = refract(I, normalize(fragNormal), refractive_index);
  // outColor = vec4(texture(cubemap, R).rgb, 1.0);

  if (material.colorTextureSet > -1)
  {
    outColor = texture(textures[material.colorTextureSet], fragCoords);
    if (outColor.a < 0.01) {
      discard;
    }
  } else {
    outColor = material.baseColorFactor;
  }
}
