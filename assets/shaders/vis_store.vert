#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_ARB_shader_draw_parameters : enable

#define kProjViewMatricesBindingPos 0
#define kModelMatricesBindingPos 0

layout (location = 0) in vec3 pos;

layout (location = 0) flat out uint draw_id;

layout (constant_id = 0) const uint num_materials = 1U;
layout (constant_id = 1) const uint num_lights = 1U;

layout (std430, set = 0, binding = kProjViewMatricesBindingPos)
    buffer MainStaticBuffer {
  mat4 proj;
  mat4 view;
  mat4 inv_proj;
  mat4 inv_view;
};


layout (std430, set = 1, binding = kModelMatricesBindingPos) buffer ModelMats {
  mat4 model_mats[];
};

void main() {
  draw_id = gl_DrawIDARB;
  gl_Position = proj * view * model_mats[draw_id] * vec4(pos, 1.f);
}
