#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_ARB_shader_image_load_store : enable

#define kProjViewMatricesBindingPos 0
#define kModelMatricesBindingPos 0
#define kLightsArrayBindingPos 10
#define kMatConstsArrayBindingPos 9
#define kIndirectDrawCmdsBindingPos 3
#define kVertexBufferBindingPos 4
#define kIndexBufferBindingPos 2
#define kMaterialIDsBindingPos 1
#define kDepthBuffBindingPos 1
#define kDiffuseTexturesArrayBindingPos 2
#define kAmbientTexturesArrayBindingPos 3
#define kSpecularTexturesArrayBindingPos 4
#define kNormalTexturesArrayBindingPos 5
#define kRoughnessTexturesArrayBindingPos 6
#define kVisBufferBindingPos 7
#define kDerivsBarysBufferBindingPos 11

struct VkDrawIndexedIndirectCommand {
  uint indexCount;
  uint instanceCount;
  uint firstIndex;
  int vertexOffset;
  uint firstInstance;
};

struct Light {
  vec4 pos_radius;
  vec4 diff_colour;
  vec4 spec_colour;
};

// Could be packed better but it's kept like this until optimisation stage
struct MatConsts {
  vec4 diffuse_dissolve;
  vec4 specular_shininess;
  vec4 ambient;
  /* 32-bit padding goes here on host side, but GLSL will transform
     the ambient vec3 into a vec4 */
  vec4 emission;
};

struct Vertex {
  vec3 pos;
  vec3 normal;
  vec2 uv;
};


layout (constant_id = 0) const uint num_materials = 1U;
layout (constant_id = 1) const uint num_lights = 1U;


layout (std430, set = 0, binding = kProjViewMatricesBindingPos)
    buffer MainStaticBuffer {
  mat4 proj;
  mat4 view;
  mat4 inv_proj;
  mat4 inv_view;
};
  
layout (std430, set = 0, binding = kLightsArrayBindingPos)
    buffer Lights {
  Light lights[num_lights];
};

layout (std430, set = 0, binding = kMatConstsArrayBindingPos)
    buffer MatConstsArray {
  MatConsts mat_consts[num_materials];
};

layout (set = 0, binding = kDepthBuffBindingPos) uniform
  sampler2D depth_buffer;

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

layout (set = 0, binding = kVisBufferBindingPos)
  uniform usampler2D vis_buff;

layout (set = 0, binding = kDerivsBarysBufferBindingPos)
  uniform usampler2D derivs_and_barys_buff;

layout (location = 0) in vec2 screen_pos;
layout (location = 1) in vec3 view_ray;
layout (location = 0) out vec4 col;

layout (std430, set = 1, binding = kIndirectDrawCmdsBindingPos)
    buffer IndirectDraws {
  VkDrawIndexedIndirectCommand indirect_draws[];
};

struct Vec3 {
  float x, y, z;
};

struct Vec2 {
  float x, y;
};

layout (std430, set = 1, binding = kVertexBufferBindingPos) buffer VtxPos {
  Vec3 vtx_pos[];
};

layout (std430, set = 1, binding = kVertexBufferBindingPos + 1) buffer Normal {
  Vec3 normals[];
};

layout (std430, set = 1, binding = kVertexBufferBindingPos + 2) buffer UVsf {
  Vec2 uvs[];
};

layout (std430, set = 1, binding = kVertexBufferBindingPos + 3) buffer Tang {
  Vec3 tangents[];
};

layout (std430, set = 1, binding = kVertexBufferBindingPos + 4) buffer Bitang {
  Vec3 bitangents[];
};

layout (std430, set = 1, binding = kIndexBufferBindingPos) buffer IdxBuff {
  uint idx_buff[];
};

layout (std430, set = 1, binding = kModelMatricesBindingPos) buffer ModelMats {
  mat4 model_mats[];
};

layout (std430, set = 1, binding = kMaterialIDsBindingPos) buffer MatIDs {
  uint mat_ids[];
};

void ComputeBaryDerivatives(in vec2 pos_scr[3], out vec3 db_dx,
                            out vec3 db_dy, out float det) {
  det = determinant(mat2x2(
       pos_scr[0] - pos_scr[1], pos_scr[2] - pos_scr[1]));
  float d = 1.0f / det;
  db_dx = vec3(
      pos_scr[2].y - pos_scr[1].y,
      pos_scr[0].y - pos_scr[2].y,
      pos_scr[1].y - pos_scr[0].y) * d;
  db_dy = vec3(
      pos_scr[1].x - pos_scr[2].x,
      pos_scr[2].x - pos_scr[0].x,
      pos_scr[0].x - pos_scr[1].x) * d;
}

// The InterpAttributes methods assume that the gl_BaryCoordSmoothAMD coords
// are the linear barycentric coordinates multiplied by the depth at the
// pixel location.
vec3 InterpAttributes(in mat3x3 attributes, in vec3 bary_coords) {
  return attributes * bary_coords;
}

vec2 InterpAttributes(in mat3x2 attributes, in vec3 bary_coords) {
  return attributes * bary_coords;
}

vec3 CalcLighting(
    in vec3 normal,
    in vec3 position,
    in vec3 diff_albedo,
    in vec3 ambient_albedo,
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
    vec3 specular = pow(max(dot(normal, H), 0.f), spec_power) *
      lights[i].spec_colour.rgb * spec_albedo * nDotL;

    colour = colour + ambient_albedo + ((specular + diffuse) *
             vec3(attenuation));
  }

  return colour;
}

// Use + instead of - because a RH proj matrix forced
// between 0 and 1 would keep these coordinates negative and
// proj_mat[2][2] would be negative instead of positive.
float LineariseDepth(in float depth, in mat4 proj_mat) {
  return proj_mat[3][2] / (depth + proj_mat[2][2]);
}

void main() {
  uvec2 vis_and_barys_raw = texelFetch(vis_buff, ivec2(gl_FragCoord.xy), 0).rg;
  uint vis_raw = vis_and_barys_raw.x;

  if(vis_raw != 0) {
    // Unpack data from the vis buffer
    uint draw_id = (vis_raw >> 23) & 0x000000FF;
    uint triangle_id = vis_raw & 0x007FFFFF;
    uint alpha = vis_raw >> 31;

    // Retrieve the triangle's vertex indices 
    uint start_idx = indirect_draws[draw_id].firstIndex;
    uint tri_id_0 = (triangle_id * 3 + 0) + start_idx;
    uint tri_id_1 = (triangle_id * 3 + 1) + start_idx;
    uint tri_id_2 = (triangle_id * 3 + 2) + start_idx;

    // TODO use filtered and culled index buffer to read indices
    uint idx_0 = idx_buff[tri_id_0];
    uint idx_1 = idx_buff[tri_id_1];
    uint idx_2 = idx_buff[tri_id_2];

    // Read vertices
    /*vec4 vtx0_pos = vec4(vtx_pos[idx_0].x, vtx_pos[idx_0].y,
                         vtx_pos[idx_0].z, 1.f);
    vec4 vtx1_pos = vec4(vtx_pos[idx_1].x, vtx_pos[idx_1].y,
                         vtx_pos[idx_1].z, 1.f);
    vec4 vtx2_pos = vec4(vtx_pos[idx_2].x, vtx_pos[idx_2].y,
                         vtx_pos[idx_2].z, 1.f);
    // Tranform vertices to clip space
    vtx0_pos = proj * view * vtx0_pos;
    vtx1_pos = proj * view * vtx1_pos;
    vtx2_pos = proj * view * vtx2_pos;

    // Cache inverse of w component of positions, the view space z position
    vec3 one_over_w = 1.f / vec3(vtx0_pos.w, vtx1_pos.w, vtx2_pos.w);*/

    /*// Transform vertices to screen space*/
    /*vec2 pos_scr[3] = { vtx0_pos.xy * one_over_w.x, vtx1_pos.xy * one_over_w.y,*/
                        /*vtx2_pos.xy * one_over_w.z };*/

    /*// Gradient barycentric coordinates for x and y*/
    /*vec3 db_dx, db_dy;*/
    /*float det;*/
    /*ComputeBaryDerivatives(pos_scr, db_dx, db_dy, det);*/

    /*// Get the displacement vector from the vtx pos to the pixel pos*/
    /*// in screen space*/
    /*vec2 d = screen_pos - pos_scr[0];*/

    /*// Calculate the w component at the pixel location; the w component*/
    /*// represents the z value in view space*/
    /*float w = 1.0 / InterpAttributes(one_over_w, db_dx, db_dy, d);*/

    // Calculate position in view space from the depth buffer
    ivec2 sample_idx = ivec2(gl_FragCoord.xy);
    float depth = texelFetch(depth_buffer, sample_idx, 0).r;
    float linear_depth = LineariseDepth(depth, proj);
    vec3 position = view_ray * linear_depth;

    // Retrive data from the derivatives and bary coords buffers
    uvec2 derivs_and_barys_buff_data =
      texelFetch(derivs_and_barys_buff, sample_idx, 0).xy;
    vec2 bary_coords_unpacked = unpackUnorm2x16(vis_and_barys_raw.y);
    vec3 bary_coords = vec3(
        bary_coords_unpacked,
        1.f - bary_coords_unpacked.x - bary_coords_unpacked.y);

    vec2 dfdx = unpackSnorm2x16(derivs_and_barys_buff_data.x);
    vec2 dfdy = unpackSnorm2x16(derivs_and_barys_buff_data.y);

    // Interpolate the texture coordinates
    mat3x2 tex_coords_tri = {
      vec2(uvs[idx_0].x, uvs[idx_0].y),
      vec2(uvs[idx_1].x, uvs[idx_1].y),
      vec2(uvs[idx_2].x, uvs[idx_2].y),
    };
    vec2 tex_coords = InterpAttributes(tex_coords_tri, bary_coords);

    // Interpolate the normals
    mat3x3 normals_mat = {
      vec3(normals[idx_0].x, normals[idx_0].y, normals[idx_0].z),
      vec3(normals[idx_1].x, normals[idx_1].y, normals[idx_1].z),
      vec3(normals[idx_2].x, normals[idx_2].y, normals[idx_2].z),
    };
    mat3 transp_model_view = transpose(inverse(mat3(view)));
    vec3 normal_obj = InterpAttributes(normals_mat, bary_coords);

    vec3 norm_vs = (transp_model_view * normal_obj);

    // Interpolate the tangents 
    mat3x3 tangents_mat = {
      vec3(tangents[idx_0].x, tangents[idx_0].y, tangents[idx_0].z),
      vec3(tangents[idx_1].x, tangents[idx_1].y, tangents[idx_1].z),
      vec3(tangents[idx_2].x, tangents[idx_2].y, tangents[idx_2].z),
    };
    vec3 tangent_obj = InterpAttributes(tangents_mat, bary_coords);
    vec3 tangent_vs = normalize(transp_model_view * tangent_obj);

    // Interpolate the bitangents
    mat3x3 bitangents_mat = {
      vec3(bitangents[idx_0].x, bitangents[idx_0].y, bitangents[idx_0].z),
      vec3(bitangents[idx_1].x, bitangents[idx_1].y, bitangents[idx_1].z),
      vec3(bitangents[idx_2].x, bitangents[idx_2].y, bitangents[idx_2].z),
    };
    vec3 bitangent_obj = InterpAttributes(bitangents_mat, bary_coords);
    vec3 bitangent_vs = normalize(transp_model_view * bitangent_obj);

    mat3 tangent_frame_vs = mat3(
      normalize(tangent_vs),
      normalize(bitangent_vs),
      normalize(norm_vs));

    //tex_coords.y = 1- tex_coords.y;
    /* Sample the tangent space normal map */
    uint mat_id = mat_ids[draw_id];
    /* vec3 normal_ts = textureGrad(norm_textures[mat_id], tex_coords,
                                 dfdx, dfdy).rgb;*/
    vec3 normal_ts = texture(norm_textures[mat_id], tex_coords).rgb;
    normal_ts = normalize((normal_ts * 2.f) - 1.f);

    vec3 normal_vs = vec3(tangent_frame_vs * normal_ts);

    // Get diffuse albedo from the map
    float gamma = 2.2f;
    vec3 diff_albedo =
      pow(textureGrad(diff_textures[mat_id], tex_coords,
                      dfdx, dfdy).rgb, vec3(gamma)) *
          mat_consts[mat_id].diffuse_dissolve.rgb;
    vec3 spec_albedo =
      pow(textureGrad(spec_textures[mat_id], tex_coords,
                      dfdx, dfdy).rgb, vec3(gamma)) *
      mat_consts[mat_id].specular_shininess.rgb;
    float spec_power =
      pow(textureGrad(rough_textures[mat_id], tex_coords,
                  dfdx, dfdy).rgb, vec3(gamma)).r *
      mat_consts[mat_id].specular_shininess.a;
    vec3 ambient_albedo =
      pow(textureGrad(amb_textures[mat_id], tex_coords,
                      dfdx, dfdy).rgb, vec3(gamma)) *
      mat_consts[mat_id].ambient.rgb;

/*    col = vec4(
        CalcLighting(normal_vs, position, diff_albedo,
            ambient_albedo, spec_albedo, spec_power),
        1.f);*/
      col = vec4(normal_vs, float(draw_id) / 40.f);
  }
}
