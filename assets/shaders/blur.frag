#version 440

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define kSSAOBuffersBindingPos 8

layout (location = 1) in vec2 in_uv;

layout (location = 0) out float blur_result;

layout (constant_id = 0) const uint kSSAONoiseTextureSize = 4U;

layout (set = 0, binding = kSSAOBuffersBindingPos) uniform
  sampler2D[2] SSAO_buffers;

void main() {
	int blur_size = int(kSSAONoiseTextureSize * kSSAONoiseTextureSize);
	float accumulation = 0.f;
	
  vec2 texel_size = 1.f / vec2(textureSize(SSAO_buffers[0], 0));
	vec2 hlim = vec2(((-float(kSSAONoiseTextureSize)) * 0.5f) + 0.5f);

	for (int x = 0; x < kSSAONoiseTextureSize; x++) {
		for (int y = 0; y < kSSAONoiseTextureSize; y++) {
			vec2 offset = (vec2(float(x), float(y)) + hlim) * texel_size;
			accumulation += texture(SSAO_buffers[0], in_uv + offset).r;
		}
	}

	blur_result = accumulation / (float(blur_size));
}
