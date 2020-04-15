#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 vert_texcoord;

layout(binding = 1) uniform samplerCube environmentMap;

layout(location = 0) out vec4 outColor;

void main()
{
  vec3 envColor = texture(environmentMap, vert_texcoord).rgb;

  envColor = envColor / (envColor + vec3(1.0));
  envColor = pow(envColor, vec3(1.0/2.2));

  outColor = vec4(envColor, 1.0);
}
