#version 440

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

/* TODO Use either UBO or spec. constants to set lights num const */
#define kLightsNum 2

#define kGBufferBaseBindingPos 12
#define kProjViewMatricesBindingPos 0
#define kLightsArrayBindingPos 10
#define kIndirectDrawCmdsBindingPos 2
#define kDepthBufferBindingPos 1
#define kDiffuseTexturesArrayBindingPos 2
#define kAmbientTexturesArrayBindingPos 3
#define kSpecularTexturesArrayBindingPos 4
#define kNormalTexturesArrayBindingPos 5
#define kRoughnessTexturesArrayBindingPos 6
#define kSSAOBuffersBindingPos 8

layout (location = 0) in vec3 view_ray;

layout (location = 0) out vec4 hdr_colour;


struct VkDrawIndexedIndirectCommand {
  uint indexCount;
  uint instanceCount;
  uint firstIndex;
  int vertexOffset;
  uint firstInstance;
};

layout (std430, set = 0, binding = kProjViewMatricesBindingPos)
    buffer ProjView {
  mat4 proj;
  mat4 view;
  mat4 inv_proj;
  mat4 inv_view;
};

struct Light {
  vec4 pos_radius;
  vec4 diff_colour;
  vec4 spec_colour;
};

layout (constant_id = 0) const uint num_materials = 1U;
layout (constant_id = 1) const uint num_lights = 1U;


layout (std430, set = 0, binding = kLightsArrayBindingPos) buffer LightsArray {
  Light lights[num_lights];
};

layout (std430, set = 1, binding = kIndirectDrawCmdsBindingPos)
    buffer IndirectDraws {
  VkDrawIndexedIndirectCommand indirect_draws[];
};

layout (set = 0, binding = kDepthBufferBindingPos) uniform
  sampler2D depth_buffer;
layout (set = 0, binding = kSSAOBuffersBindingPos) uniform
  sampler2D[2] SSAO_buffers;

layout (set = 0, binding = kGBufferBaseBindingPos) uniform
  sampler2D diff_albedo_map;
layout (set = 0, binding = kGBufferBaseBindingPos + 1) uniform
  sampler2D spec_albedo_map;
layout (set = 0, binding = kGBufferBaseBindingPos + 2) uniform
  sampler2D normals_map;

// Use + instead of - because a RH proj matrix forced
// between 0 and 1 would keep these coordinates negative and
// proj_mat[2][2] would be negative instead of positive.
float LineariseDepth(in float depth, in mat4 proj_mat) {
  return proj_mat[3][2] / (depth + proj_mat[2][2]);
}

void GetGBufferAttributes(
    in vec2 screen_pos,
    out vec3 normal,
    out vec3 position,
    out vec3 diff_albedo,
    out vec3 spec_albedo,
    out float spec_power,
    out float ambient_occlusion) {
  ivec2 sample_idx = ivec2(screen_pos);

  vec4 normal_specpower = texelFetch(normals_map, sample_idx, 0);
  normal = normal_specpower.xyz;

  float depth = texelFetch(depth_buffer, sample_idx, 0).r;
  float linear_depth = LineariseDepth(depth, proj); 
  position = view_ray * linear_depth;
  
  float gamma = 2.2f;
  // The specular power exponent is encoded in the w of the normal map
  diff_albedo =
    pow(texelFetch(diff_albedo_map, sample_idx, 0).rgb, vec3(gamma));
  vec4 spec = pow(
    vec4(texelFetch(spec_albedo_map, sample_idx, 0).rgb, normal_specpower.w),
    vec4(gamma));

  ambient_occlusion = texelFetch(SSAO_buffers[0], sample_idx, 0).r;
   
  spec_albedo = spec.rgb;
  spec_power = spec.a;
}

vec3 CalcLighting(
    in vec3 normal,
    in vec3 position,
    in vec3 diff_albedo,
    in vec3 spec_albedo,
    in float spec_power,
    in float ambient_occlusion) {

  vec3 colour = vec3(0.f);

  for (uint i = 0; i < kLightsNum; i++) {
    // Calculate diffuse term of the BRDF
    vec3 L = lights[i].pos_radius.xyz - position;
    
    float dist = length(L);
    float attenuation = max(0.f, 1.f - (dist / lights[i].pos_radius.w));

    L /= dist;

    float nDotL = max(0.f, dot(normal, L));
    vec3 diffuse = diff_albedo * lights[i].diff_colour.rgb * nDotL;

    // Calculate the specular term of the BRDF
    vec3 V = normalize(-position);
    vec3 H = normalize(L + V);
    vec3 specular = pow(max(dot(normal, H), 0.f), 94.f) *
      lights[i].spec_colour.rgb * spec_albedo * nDotL; 
    
    colour = colour + ((specular + diffuse) * vec3(attenuation));

  }
    
  colour *= ambient_occlusion.rrr;

  return colour;
}

void main() {
  vec3 normal;
  vec3 position;
  vec3 diff_albedo;
  vec3 spec_albedo;
  float spec_power;
  float ambient_occlusion;

  GetGBufferAttributes(
    gl_FragCoord.xy,
    normal,
    position,
    diff_albedo,
    spec_albedo,
    spec_power,
    ambient_occlusion); 

  float attenuation = 0.f;  
  vec3 lighting = CalcLighting(
    normal,
    position,
    diff_albedo,
    spec_albedo,
    spec_power,
    ambient_occlusion);

  hdr_colour = vec4(lighting, attenuation);
}
