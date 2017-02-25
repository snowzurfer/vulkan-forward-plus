#ifndef VKS_MATERIALCONSTANTS
#define VKS_MATERIALCONSTANTS

#include <glm/glm.hpp>

namespace vks {

struct MaterialConstants {
  MaterialConstants();

  glm::vec4 diffuse_dissolve;
  glm::vec4 specular_shininess;
  glm::vec3 ambient;
	float padding;
  glm::vec3 emission;
  float padding_2;

}; // struct MaterialConstants

} // namespace vks

#endif
