#ifndef VKS_VULKANTEXTUREMANAGER
#define VKS_VULKANTEXTUREMANAGER

#include <vulkan/vulkan.h>
#include <map>
#include <vulkan_texture.h>
#include <EASTL/unique_ptr.h>
#include <EASTL/vector.h>
#include <EASTL/hash_map.h>
#include <EASTL/string.h>

namespace vks {

class VulkanDevice;

extern const eastl::string kBaseAssetsPath;

class VulkanTextureManager {
 public:
  VulkanTextureManager();

  void Init(const VulkanDevice &device);

  void Shutdown(const VulkanDevice &device);

  /**
   * @brief Load2DTexture Load a 2D texture with all its mip levels
   *
   * @param filename File to load
   * @param format Vulkan format of the data stored in the file
   * @param texture Ref to the texture object in which to load the image
   * @param VkImageUsageFlags Usage flags fo the image, defaults to
   *        VK_IMAGE_USAGE_SAMPLED_BIT
   */
  void Load2DTexture(
      const VulkanDevice &device,
      const eastl::string &filename,
      VkFormat format,
      VulkanTexture **texture,
      const VkSampler aniso_sampler,
      const VkImageUsageFlags img_flags = VK_IMAGE_USAGE_SAMPLED_BIT);
  
  void Load2DTexture(
      const VulkanDevice &device,
      const eastl::string &filename,
      VkFormat format,
      VulkanTexture **texture,
      const VkImageUsageFlags img_flags = VK_IMAGE_USAGE_SAMPLED_BIT);

  void Create2DTextureFromData(
      const VulkanDevice &device,
      const eastl::string &name,
      const void *data,
      const uint32_t size,
      const uint32_t width,
      const uint32_t height,
      VkFormat format,
      VulkanTexture **texture,
      const VkSampler aniso_sampler,
      const VkImageUsageFlags img_flags = VK_IMAGE_USAGE_SAMPLED_BIT);

  void CreateUniqueTexture(
      const VulkanDevice &device,
      VulkanTextureInitInfo &init_info,
      const eastl::string &name,
      VulkanTexture **texture);

  void Load2DPNGTexture(
      const VulkanDevice &device,
      const eastl::string &filename,
      VkFormat format,
      VulkanTexture **texture,
      const VkSampler aniso_sampler,
      const VkImageUsageFlags img_flags = VK_IMAGE_USAGE_SAMPLED_BIT);

  // Returns nullptr if texture isn't present
  VulkanTexture *GetTextureByName(const eastl::string &name);

 private:
  VkCommandBuffer cmd_buffer_;

  typedef eastl::hash_map<eastl::string,
    eastl::unique_ptr<VulkanTexture>> NameTexMap;
  NameTexMap textures_;

  void CreateTexture(
      const VulkanDevice &device,
      const eastl::string &name,
      const void *data,
      const uint32_t size,
      const uint32_t width,
      const uint32_t height,
      const uint32_t mip_levels,
      VkFormat format,
      const eastl::vector<VkBufferImageCopy> &copy_regions,
      VulkanTexture **texture,
      const VkSampler aniso_sampler,
      const VkImageUsageFlags img_flags);


}; // class VulkanTextureManager

} // namespace vks

#endif
