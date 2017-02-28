#include <material_instance.h>
#include <material.h>
#include <vulkan_device.h>
#include <utility>
#include <logger.hpp>
#include <vulkan_tools.h>
#include <vulkan_texture.h>
#include <material.h>
#include <base_system.h>
#include <vulkan_buffer.h>

namespace vks {

MaterialInstanceBuilder::MaterialInstanceBuilder(
    const eastl::string &inst_name,
    const eastl::string &mats_directory,
    const VkSampler aniso_sampler)
    : inst_name_(inst_name),
      mats_directory_(mats_directory),
      consts_(),
      textures_(),
      aniso_sampler_(aniso_sampler) {}

void MaterialInstanceBuilder::AddTexture(
    const MaterialBuilderTexture &texture_info) {
  textures_.push_back(texture_info);
}

void MaterialInstanceBuilder::AddConstants(
    const MaterialConstants &consts) {
  consts_.push_back(consts);
}

MaterialInstance::MaterialInstance()
    : name_(),
      consts_(),
      textures_({nullptr}),
      material_(nullptr),
      maps_desc_set_(VK_NULL_HANDLE) {}

void MaterialInstance::Init(
    const VulkanDevice &device,
    const MaterialInstanceBuilder &builder) {
  consts_ = builder.consts().front();

  uint32_t builder_textures_count = SCAST_U32(builder.textures().size());
  std::vector<VkWriteDescriptorSet> set_writes(builder_textures_count);
  for (uint32_t i = 0U; i < builder_textures_count; i++) {
    VulkanTexture *loaded_texture = nullptr;
    if (builder.textures()[i].name != "") {
      if (builder.textures()[i].name.find("png") != -1) {
        texture_manager()->Load2DPNGTexture(
          device,
          builder.mats_directory() + builder.textures()[i].name,
          VK_FORMAT_R8G8B8A8_UNORM,
          &loaded_texture,
          builder.aniso_sampler());
      }
      else if (builder.textures()[i].name.find("dds") != -1) {
        texture_manager()->Load2DTexture(
          device,
          builder.mats_directory() + builder.textures()[i].name,
          VK_FORMAT_BC2_UNORM_BLOCK,
          &loaded_texture,
          builder.aniso_sampler());
      }
    }
    if (loaded_texture == nullptr) {
      loaded_texture = texture_manager()->GetTextureByName(
          STR(ASSETS_FOLDER) "dummy.ktx");
    }

    textures_[tools::ToUnderlying(builder.textures()[i].type)] = loaded_texture;
  }

  // Check if some of the textures haven't been assigned
  uint32_t textures_count = SCAST_U32(textures_.size());
  for (uint32_t i = 0U; i < textures_count; i++) {
    if (textures_[i] == nullptr) {
      textures_[i] = texture_manager()->GetTextureByName(
          STR(ASSETS_FOLDER) "dummy.ktx");
    }
  }

  name_ = builder.inst_name();

  LOG("Finished init of MatInstance " + name_);
}

void MaterialInstance::Shutdown(const VulkanDevice &device) {
  LOG("Shutdown matinstance " + name_);
}

} // namespace vks
