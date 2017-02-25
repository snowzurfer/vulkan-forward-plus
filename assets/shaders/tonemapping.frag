#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define kAccumBufferBindingPos 7

layout (set = 0, binding = kAccumBufferBindingPos) uniform
  sampler2D accumm_buff;

layout (location = 0) out vec4 colour;

void main() {
  const float exposure = 0.01f;

  vec3 hdr_colour = texelFetch(accumm_buff, ivec2(gl_FragCoord.xy), 0).rgb;
  float attenuation = texelFetch(accumm_buff, ivec2(gl_FragCoord.xy), 0).a;

  // Reinhard tone mapping
  vec3 mapped = vec3(1.f) - exp(-hdr_colour * exposure);

  colour = vec4(mapped, attenuation);
}
