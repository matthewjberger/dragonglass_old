#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 vert_texcoord;

layout(binding = 1) uniform samplerCube environmentMap;

layout(location = 0) out vec4 outColor;

// From http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 Uncharted2Tonemap(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	float W = 11.2;
	return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

vec4 tonemap(vec4 color)
{
	vec3 outcol = Uncharted2Tonemap(color.rgb);
	outcol = outcol * (1.0f / Uncharted2Tonemap(vec3(11.2f)));
	return vec4(pow(outcol, vec3(1.0f / 2.2)), color.a);
}

vec4 SRGBtoLINEAR(vec4 srgbIn)
{
	vec3 linOut = pow(srgbIn.xyz,vec3(2.2));
	return vec4(linOut,srgbIn.w);;
}

void main()
{
	vec3 color = SRGBtoLINEAR(tonemap(textureLod(environmentMap, vert_texcoord, 1.5))).rgb;
	outColor = vec4(color * 1.0, 1.0);
}
