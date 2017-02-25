#ifndef VKS_VULKANUNIFORMBUFFER
#define VKS_VULKANUNIFORMBUFFER

#include <vulkan_buffer.h>
#include <string>

namespace vks {

class VulkanUniformBuffer : public VulkanBuffer {
 public:
  VulkanUniformBuffer();
  ~VulkanUniformBuffer();

  void set_name(const std::string &name) { name_ = name; }
  const std::string &name() const { return name_; }

 private:
  std::string name_;

}; // class VulkanUniformBuffer

} // namespace vks

#endif
