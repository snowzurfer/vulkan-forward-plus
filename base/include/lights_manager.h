#ifndef VKS_LIGHTS_MANAGER
#define VKS_LIGHTS_MANAGER

#include <light.h>
#include <EASTL/vector.h>
#include <glm/mat4x4.hpp>

namespace vks {

class LightsManager {
 public:
  LightsManager();

  Light *CreateLight(const glm::vec3 &diffuse, const glm::vec3 &specular,
                     const glm::vec3 &position, float radius);

  const eastl::vector<Light> &lights() const { return lights_; }
  uint32_t GetNumLights() const;

  eastl::vector<Light> TransformLights(const glm::mat4 &transform);

 private:
  eastl::vector<Light> lights_;

}; // class LightsManager

} // namespace vks

#endif
