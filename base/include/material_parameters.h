#ifndef VKS_MATERIALPARAMETERS
#define VKS_MATERIALPARAMETERS

#include <vector>
#include <vulkan_uniform_buffer.h>
#include <material_constants.h>
#include <vulkan/vulkan.h>

namespace vks {

class MaterialParameters {
 public:
  MaterialParameters();


 private:
  MaterialConstants consts_;

  std::string ambient_tex;            // map_Ka
  std::string diffuse_tex;            // map_Kd
  std::string specular_tex;           // map_Ks
  std::string specular_highlight_tex; // map_Ns
  std::string bump_tex;               // map_bump, bump
  std::string displacement_tex;       // disp
  std::string alpha_tex;              // map_d

}; // class MaterialParameters

} // namespace vks

#endif
