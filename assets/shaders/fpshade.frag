#version 440

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define kProjViewMatricesBindingPos 0
#define kDiffuseTexturesArrayBindingPos 2
#define kAmbientTexturesArrayBindingPos 3
#define kSpecularTexturesArrayBindingPos 4
#define kNormalTexturesArrayBindingPos 5
#define kRoughnessTexturesArrayBindingPos 6
#define kLightsArrayBindingPos 8
#define kLightsIdxsBindingPos 9
#define kLightsGridBindingPos 10
#define kMatConstsArrayBindingPos 11

#define kModelMatricesBindingPos 0
#define kMaterialIDsBindingPos 1

layout (location = 0) in vec3 pos_vs;
layout (location = 1) in vec3 norm_vs;
layout (location = 2) in vec3 uv_fs;
layout (location = 3) in vec3 bitangent_vs;
layout (location = 4) in vec3 tangent_vs;
layout (location = 5) flat in uint mesh_id;

layout (location = 0) out vec4 hdr_colour;

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

// Could be packed better but it's kept like this until optimisation stage
struct MatConsts {
  vec4 diffuse_dissolve;
  vec4 specular_shininess;
  vec4 ambient;
  /* 32-bit padding goes here on host side, but GLSL will transform
     the ambient vec3 into a vec4 */
  vec4 emission;
};

layout (std430, set = 0, binding = kMatConstsArrayBindingPos)
    buffer MatConstsArray {
  MatConsts mat_consts[num_materials];
};

layout (set = 0, binding = kDiffuseTexturesArrayBindingPos)
  uniform sampler2D[num_materials] diff_textures;
layout (set = 0, binding = kAmbientTexturesArrayBindingPos)
  uniform sampler2D[num_materials] amb_textures;
layout (set = 0, binding = kSpecularTexturesArrayBindingPos)
  uniform sampler2D[num_materials] spec_textures;
layout (set = 0, binding = kNormalTexturesArrayBindingPos)
  uniform sampler2D[num_materials] norm_textures;
layout (set = 0, binding = kRoughnessTexturesArrayBindingPos)
  uniform sampler2D[num_materials] rough_textures;

layout (std430, set = 1, binding = kMaterialIDsBindingPos) buffer MatIDs {
  uint mat_ids[];
};

layout (std430, set = 0, binding = kLightsArrayBindingPos) buffer LightsArray {
  Light lights[num_lights];
};

layout (std430, set = 0, binding = kLightsIdxsBindingPos) buffer LightsIdxs {
  uint lights_idxs[];
};

layout (std430, set = 0, binding = kLightsGridBindingPos) buffer LightsGrid {
  uint lights_grid[];
};

void GetAttributes(
    out vec3 normal,
    out vec3 diff_albedo,
    out vec3 spec_albedo,
    out float spec_power) {
  uint mat_id = mat_ids[mesh_id];
  diff_albedo =
    texture(diff_textures[mat_id], uv_fs.xy).rgb *
      mat_consts[mat_id].diffuse_dissolve.rgb;

  mat3 tangent_frame_vs = mat3(
    normalize(tangent_vs),
    normalize(bitangent_vs),
    normalize(norm_vs));

  /* Sample the tangent space normal map */
  vec3 normal_ts = texture(norm_textures[mat_id], uv_fs.xy).rgb;
  normal_ts = normalize((normal_ts * 2.f) - 1.f);
  normal = tangent_frame_vs * normal_ts;

  spec_albedo =
    texture(spec_textures[mat_id], uv_fs.xy).rgb *
      mat_consts[mat_id].specular_shininess.rgb;

  spec_power = texture(rough_textures[mat_id], uv_fs.xy).r *
        mat_consts[mat_id].specular_shininess.a;
}


vec3 CalcLighting(
    in vec3 normal,
    in vec3 position,
    in vec3 diff_albedo,
    in vec3 spec_albedo,
    in float spec_power) {
  vec3 colour = vec3(0.f);

  for (uint i = 0; i < num_lights; i++) {
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

  return colour;
}

void main() {
  vec3 normal;
  vec3 diff_albedo;
  vec3 spec_albedo;
  float spec_power;

  GetAttributes(
    normal,
    diff_albedo,
    spec_albedo,
    spec_power);

  float attenuation = 0.f;
  vec3 lighting = CalcLighting(
    normal,
    pos_vs,
    diff_albedo,
    spec_albedo,
    spec_power);

  hdr_colour = vec4(lighting, attenuation);
}
