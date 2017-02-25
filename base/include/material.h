#ifndef VKS_MATERIAL
#define VKS_MATERIAL

#include <EASTL/string.h>
#include <vulkan/vulkan.h>
#include <material_constants.h>
#include <EASTL/vector.h>
#include <cstdint>
#include <EASTL/array.h>
#include <model.h>
#include <EASTL/unique_ptr.h>
#include <viewport.h>
#include <shaderc/shaderc.h>
#include <vertex_setup.h>

namespace vks {

class VulkanDevice;

extern const eastl::string kBaseShaderAssetsPath;

enum class ShaderTypes : uint8_t {
  VERTEX = 0U,
  TESSELLATION_CONTROL,
  TESSELLATION_EVALUATION,
  GEOMETRY,
  FRAGMENT,
  COMPUTE,
  count 
}; // enum class ShaderTypes

class MaterialShader {
 public:
  MaterialShader(
      const eastl::string &file_name,
      const eastl::string &entry_point,
      const ShaderTypes &type);

  void AddSpecialisationEntry(
    uint32_t    constant_id,
    uint32_t    size,
    const void *data);

  void SetSpecialisation(const VkSpecializationInfo &info);
  void SetSpecialisation(VkSpecializationInfo &&info);

  const VkSpecializationInfo *spec_info() const {
    return &spec_info_;
  }
  const eastl::string &file_name() const { return file_name_; }
  const eastl::string &entry_point() const { return entry_point_; }
  const ShaderTypes type() const { return type_; }

  VkPipelineShaderStageCreateInfo Compile(
      const VulkanDevice &device,
      const shaderc_compiler_t compiler);

  void ShutdownModule(const VulkanDevice &device);

 private:
  eastl::string file_name_;
  eastl::string entry_point_;
  VkSpecializationInfo spec_info_;
  eastl::vector<VkSpecializationMapEntry> info_entries_;
  eastl::vector<uint8_t> infos_data_;
  ShaderTypes type_;
  bool compiled_once_;
  VkPipelineShaderStageCreateInfo current_stage_create_info_;

  const VkShaderStageFlagBits GetVkShaderType() const;
  const shaderc_shader_kind GetShadercShaderKind() const; 


}; // class MaterialShader

class MaterialBuilder {
 public:
  MaterialBuilder(
      const VertexSetup &vertex_setup,
      const eastl::string mat_name,
      VkPipelineLayout pipe_layout,
      VkRenderPass render_pass,
      VkFrontFace front_face,
      uint32_t subpass_idx,
      const szt::Viewport &viewport);

  void GetVertexInputBindingDescription(
      eastl::vector<VkVertexInputBindingDescription> &bindings) const;
  void GetVertexInputAttributeDescriptors(
      eastl::vector<VkVertexInputAttributeDescription> &attributes) const; 
  void AddShader(eastl::unique_ptr<MaterialShader> shader);

  void AddColorBlendAttachment(
    VkBool32                 blend_enable,
    VkBlendFactor            src_color_blend_factor,
    VkBlendFactor            dst_color_blend_factor,
    VkBlendOp                color_blend_op,
    VkBlendFactor            src_alpha_blend_factor,
    VkBlendFactor            dst_alpha_blend_factor,
    VkBlendOp                alpha_blend_op,
    VkColorComponentFlags    color_write_mask);

  void AddColorBlendStateCreateInfo(
    VkBool32                                      logic_op_enable,
    VkLogicOp                                     logic_op,
    float                                         blend_constants[4U]);

  const eastl::string &mat_name() const { return mat_name_; }
  uint32_t vertex_size() const { return vertex_size_; }
  const eastl::vector<eastl::unique_ptr<MaterialShader>> &shaders() const {
    return shaders_;
  }

  void SetDepthWriteEnable(VkBool32 enable);
  void SetDepthTestEnable(VkBool32 enable);
  VkBool32 depth_test_enable() const { return depth_test_enable_; }
  VkBool32 depth_write_enable() const { return depth_write_enable_; }
  VkPipelineLayout pipe_layout() const { return pipe_layout_; }
  VkFrontFace front_face() const { return front_face_; }
  VkRenderPass render_pass() const { return render_pass_; }
  uint32_t subpass_idx() const { return subpass_idx_; }
  const VkPipelineColorBlendStateCreateInfo color_blend_state_create_info() const {
    return color_blend_state_create_info_;
  }
  const szt::Viewport &viewport() const { return viewport_; }

 private:
  eastl::vector<eastl::unique_ptr<MaterialShader>> shaders_;
  eastl::string mat_name_;
  uint32_t vertex_size_;
  VkBool32 depth_test_enable_;
  VkBool32 depth_write_enable_;
  VkPipelineLayout pipe_layout_;
  VkFrontFace front_face_;
  VkRenderPass render_pass_;
  uint32_t subpass_idx_;
  VkPipelineColorBlendStateCreateInfo color_blend_state_create_info_;
  eastl::unique_ptr<VertexSetup> vertex_setup_;
  const szt::Viewport viewport_;
  eastl::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments_;
  eastl::array<float, 4U> blend_constants_;

}; // class MaterialBuilder 

class Material {
 public:
  Material();

  void Init(const eastl::string &name);
  void Shutdown(const VulkanDevice &device);

  void InitPipeline(
      const VulkanDevice &device,
      eastl::unique_ptr<MaterialBuilder> builder);
  const VkPipeline &pipeline() const { return pipeline_; }
  const eastl::string &name() const { return name_; }

  void BindPipeline(VkCommandBuffer cmd_buff,
                    VkPipelineBindPoint bind_point) const;

  void Reload(const VulkanDevice &device);

 private:

  void CacheBuilder(eastl::unique_ptr<MaterialBuilder> builder);
  void CompileShaders(
      const VulkanDevice &device,
      eastl::vector<VkPipelineShaderStageCreateInfo> &stage_creates_out);
  void CreatePipeline(
      const VulkanDevice &device,
      eastl::vector<VkPipelineShaderStageCreateInfo> &stage_create_infos);

  eastl::string name_;
  // The pipeline as defined by the shaders of this material
  VkPipeline pipeline_;
  eastl::array<VkShaderModule, 6U> modules_;
  eastl::unique_ptr<MaterialBuilder> builder_;

  void ShutdownPipeline(const VulkanDevice &device);

}; // class Material

} // namespace vks

#endif
