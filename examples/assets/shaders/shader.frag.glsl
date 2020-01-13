#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragCoords;

layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 projection;
  vec3 camera_position;
  float shininess;
} ubo;
layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

struct DirectionalLight {
  vec3 direction;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
} directional_light;

void main() {
  directional_light.direction = -1 * vec3(0.2, 1.0, 0.3);
  directional_light.ambient = vec3(0.5, 0.5, 0.5);
  directional_light.diffuse = vec3(0.4, 0.4, 0.4);
  directional_light.specular = vec3(0.5, 0.5, 0.5);

  vec3 normal = normalize(fragNormal);
  vec3 view_direction = normalize(ubo.camera_position - fragPosition);

  vec3 directional_light_dir = normalize(-directional_light.direction);
  vec3 halfway_dir = normalize(directional_light_dir + view_direction);

  // diffuse
  float diff = max(dot(normal, directional_light_dir), 0.0);

  // specular
  vec3 reflect_dir = reflect(-directional_light_dir, normal);
  float spec = pow(max(dot(normal, halfway_dir), 0.0), ubo.shininess);

  vec3 ambient = directional_light.ambient * vec3(texture(texSampler, fragCoords));
  vec3 diffuse = directional_light.diffuse * diff * vec3(texture(texSampler, fragCoords));
  vec3 specular = directional_light.specular * spec; // * vec3(texture(material.specular_texture, fragCoords));

  vec3 result = ambient + diffuse + specular;
  outColor = vec4(result, 1.0);
}
