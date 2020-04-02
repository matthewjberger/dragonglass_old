#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragCoords_0;
layout(location = 2) in vec3 fragPosition;
layout(location = 3) in vec3 fragCameraPosition;

layout(binding = 2) uniform sampler2D textures[100];
layout(binding = 3) uniform samplerCube cubemap;

layout(push_constant) uniform Material {
  vec4 baseColorFactor;
  vec3 emissiveFactor;
  int colorTextureSet;
  int metallicRoughnessTextureSet;
  int normalTextureSet;
  int occlusionTextureSet;
  int emissiveTextureSet;
  float metallicFactor;
  float roughnessFactor;
  float alphaMask;
  float alphaMaskCutoff;
} material;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

vec3 getNormalFromMap()
{
  if (material.normalTextureSet <= -1)
    {
      return fragNormal;
    }

  vec3 tangentNormal = texture(textures[material.normalTextureSet], fragCoords_0).xyz * 2.0 - 1.0;

  vec3 Q1  = dFdx(fragPosition);
  vec3 Q2  = dFdy(fragPosition);
  vec2 st1 = dFdx(fragCoords_0);
  vec2 st2 = dFdy(fragCoords_0);

  vec3 N   = normalize(fragNormal);
  vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
  vec3 B  = -normalize(cross(N, T));
  mat3 TBN = mat3(T, B, N);

  return normalize(TBN * tangentNormal);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
  vec3 lightPositions[2] = vec3[2](vec3(1.0, -1.0, 1.0),
                                   vec3(-8.0, -1.0, 0.0));

  vec3 lightColors[2] = vec3[2](vec3(10.0, 10.0, 10.0),
                                vec3(10.0, 10.0, 10.0));

  vec3 albedo = material.baseColorFactor.xyz;
  float baseColorAlpha = material.baseColorFactor.w;
  if (material.colorTextureSet > -1)
    {
      vec4 albedoMap = texture(textures[material.colorTextureSet], fragCoords_0);
      baseColorAlpha = albedoMap.a;
      albedo = pow(albedoMap.rgb, vec3(2.2));
    }

  if (baseColorAlpha < 0.005) {
    discard;
  }

  float metallic = 1.0;
  float roughness = 1.0;
  if (material.metallicRoughnessTextureSet > -1)
    {
      vec4 physicalDescriptor = texture(textures[material.metallicRoughnessTextureSet], fragCoords_0);
      metallic = physicalDescriptor.b * material.metallicFactor;
      roughness = physicalDescriptor.g * material.roughnessFactor;
    }

  float ao = 1.0;
  if (material.occlusionTextureSet > -1)
    {
      vec4 occlusionTexture = texture(textures[material.occlusionTextureSet], fragCoords_0);
      ao = occlusionTexture.r;
    }

  vec3 N = getNormalFromMap();
  vec3 V = normalize(fragCameraPosition - fragPosition);

  // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
  // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
  vec3 F0 = vec3(0.04);
  F0 = mix(F0, albedo, metallic);

  // reflectance equation
  vec3 Lo = vec3(0.0);
  for(int i = 0; i < 2; ++i)
    {
      // calculate per-light radiance
      vec3 L = normalize(lightPositions[i] - fragPosition);
      vec3 H = normalize(V + L);
      float distance = length(lightPositions[i] - fragPosition);
      float attenuation = 1.0 / (distance * distance);
      vec3 radiance = lightColors[i] * attenuation;

      // Cook-Torrance BRDF
      float NDF = DistributionGGX(N, H, roughness);
      float G   = GeometrySmith(N, V, L, roughness);
      vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

      vec3 nominator    = NDF * G * F;
      float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; // 0.001 to prevent divide by zero.
      vec3 specular = nominator / denominator;

      // kS is equal to Fresnel
      vec3 kS = F;
      // for energy conservation, the diffuse and specular light can't
      // be above 1.0 (unless the surface emits light); to preserve this
      // relationship the diffuse component (kD) should equal 1.0 - kS.
      vec3 kD = vec3(1.0) - kS;
      // multiply kD by the inverse metalness such that only non-metals
      // have diffuse lighting, or a linear blend if partly metal (pure metals
      // have no diffuse light).
      kD *= 1.0 - metallic;

      // scale light by NdotL
      float NdotL = max(dot(N, L), 0.0);

      // add to outgoing radiance Lo
      Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }

  // TODO: Replace this ambient lighting with IBL, using an irradiance map
  vec3 ambient = vec3(0.03) * albedo * ao;

  vec3 color = ambient + Lo;

  // HDR tonemapping
  color = color / (color + vec3(1.0));

  // gamma correct
  color = pow(color, vec3(1.0/2.2));

  if (material.emissiveTextureSet > -1) {
      vec4 emissiveMap = texture(textures[material.emissiveTextureSet], fragCoords_0);
      color += pow(emissiveMap.rgb, vec3(2.2)) * material.emissiveFactor;
  }

  outColor = vec4(color, baseColorAlpha);
}
