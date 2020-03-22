#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 vert_texcoord;

layout(binding = 1) uniform samplerCube cubemap;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(cubemap, vert_texcoord);
}
