#version 440

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define kProjViewMatricesBindingPos 0
#define kModelMatricesBindingPos 0

layout (location = 0) in vec3 pos;

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

layout(push_constant) uniform PushConsts {
	uint val;
} mesh_id;

void main() {
  gl_Position = proj * view * model_mats[mesh_id.val] * vec4(pos, 1.f);
}
