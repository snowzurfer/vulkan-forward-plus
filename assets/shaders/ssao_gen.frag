#version 440

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define kProjViewMatricesBindingPos 0
#define kDepthBufferBindingPos 1
#define kSSAOBuffersBindingPos 8
#define kSSAOKernelBindingPos 9
#define kGBufferBaseBindingPos 12

layout (location = 0) in vec3 view_ray;
layout (location = 1) in vec2 in_uv;

layout (location = 0) out float occlusion;

layout (constant_id = 0) const uint kSSAOKernelSize = 64U;
layout (constant_id = 1) const float kSSAORadius = 0.5f;

layout (std430, set = 0, binding = kProjViewMatricesBindingPos)
    buffer ProjView {
  mat4 proj;
  mat4 view;
  mat4 inv_proj;
  mat4 inv_view;
};

struct Vec3 {
  float x, y, z;
};

struct Vec2 {
  float x, y;
};

layout (std430, set = 0, binding = kSSAOKernelBindingPos) buffer SSAOKernel {
  Vec2 noise_uv_scale;
  Vec3 ssao_kernel[];
};

layout (set = 0, binding = kDepthBufferBindingPos) uniform
  sampler2D depth_buffer;
layout (set = 0, binding = kSSAOBuffersBindingPos) uniform
  sampler2D[2] SSAO_buffers;
layout (set = 0, binding = kGBufferBaseBindingPos + 2) uniform
  sampler2D normals_map;


// Use + instead of - because a RH proj matrix forced
// between 0 and 1 would keep these coordinates negative and
// proj_mat[2][2] would be negative instead of positive.
float LineariseDepth(in float depth, in mat4 proj_mat) {
  return proj_mat[3][2] / (depth + proj_mat[2][2]);
}

void main() {
  ivec2 sample_idx = ivec2(gl_FragCoord.xy);

	// Get position of the fragment in view-space and its normal
  float depth = texelFetch(depth_buffer, sample_idx, 0).r;
  float linear_depth = LineariseDepth(depth, proj); 
  vec3 position = view_ray * linear_depth;
  vec3 normal = texelFetch(normals_map, sample_idx, 0).rgb;

	// Get a random vector using a noise lookup
  vec2 noise_uv_scale_vec2 = vec2(noise_uv_scale.x, noise_uv_scale.y);
	vec3 random_vec = vec3(texture(SSAO_buffers[1], in_uv * noise_uv_scale_vec2).xy, 0.f);

	// Create TBN matrix
	vec3 tangent = normalize(random_vec - normal * dot(random_vec, normal));
	vec3 bitangent = cross(tangent, normal);
	mat3 TBN = mat3(tangent, bitangent, normal);

	// Calculate occlusion value
	occlusion = 0.f;
	for(int i = 0; i < kSSAOKernelSize; i++) {
    vec3 ssao_kernel_vec3 = vec3(
        ssao_kernel[i].x,
        ssao_kernel[i].y,
        ssao_kernel[i].z);
		
    vec3 sample_pos = TBN * ssao_kernel_vec3; 
		sample_pos = position + (sample_pos * kSSAORadius); 
		
		// Project the sample to screen space
		vec4 offset = vec4(sample_pos, 1.f);
		offset = proj * offset; 
		offset.xyz /= offset.w; 
		offset.xyz = offset.xyz * 0.5f + 0.5f; 

    // Sample the depth buffer at the position in screen space of the
    // SSAO kernel sample		
		float sample_depth = texture(depth_buffer, offset.xy).r;
    sample_depth = LineariseDepth(sample_depth, proj);

		// Range check
		float range_check = smoothstep(
        0.f, 1.f,
        kSSAORadius / abs(position.z + sample_depth));
		
		occlusion += range_check * step(-sample_depth, sample_pos.z);

	}

	occlusion = (occlusion / float(kSSAOKernelSize));
}
