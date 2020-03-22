#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 1) uniform samplerCube samplerCubemap;

layout (location = 0) out vec4 outColor;

void main() {
    outColor = vec4(0.0, 1.0, 0.0, 1.0);
}
