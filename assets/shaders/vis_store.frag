#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) flat in uint draw_id;

layout (location = 0) out uint id;

uint calculate_output_VBID(bool opaque, uint draw_id, uint primitive_id) {
  uint drawID_primID = ((draw_id << 23) & 0x7F800000) |
                       (primitive_id & 0x007FFFFF);
  if (opaque) {
    return drawID_primID;
  }
  else {
    return (1 << 31) | drawID_primID;
 }
}

void main() {
  /*id = unpackUnorm4x8(calculate_output_VBID(true, draw_id,
                                            gl_PrimitiveID));*/
  id = calculate_output_VBID(true, draw_id, gl_PrimitiveID);
}
