#include <material.h>
#include <vulkan_device.h>
#include <vulkan_tools.h>
#include <fstream>
#include <utility>
#include <logger.hpp>
#include <base_system.h>

namespace vks {

const eastl::string kBaseShaderAssetsPath = "../assets/shaders/";

MaterialShader::MaterialShader(
    const eastl::string &file_name,
    const eastl::string &entry_point,
    const ShaderTypes &type)
    : file_name_(file_name),
      entry_point_(entry_point),
      spec_info_(),
      info_entries_(),
      infos_data_(),
      type_(type),
      compiled_once_(false),
      current_stage_create_info_() {
  current_stage_create_info_.module = VK_NULL_HANDLE;
}

void MaterialShader::SetSpecialisation(const VkSpecializationInfo &info) {
  spec_info_ = info;
}
void MaterialShader::SetSpecialisation(VkSpecializationInfo &&info) {
  spec_info_ = eastl::move(info);
}
  
void MaterialShader::AddSpecialisationEntry(
    uint32_t    constant_id,
    uint32_t    size,
    const void *data) {
  uint32_t curr_size = SCAST_U32(infos_data_.size());
  info_entries_.push_back(tools::inits::SpecializationMapEntry(
    constant_id,
    curr_size,
    size));

  infos_data_.resize(infos_data_.size() + size);
  memcpy(infos_data_.data() + curr_size, data, size);
}

void MaterialShader::ShutdownModule(const VulkanDevice &device) {
  if (current_stage_create_info_.module != VK_NULL_HANDLE) {
    vkDestroyShaderModule(device.device(),
                          current_stage_create_info_.module, nullptr);
    current_stage_create_info_.module = VK_NULL_HANDLE;
  }
}

VkPipelineShaderStageCreateInfo MaterialShader::Compile(
    const VulkanDevice &device,
    const shaderc_compiler_t compiler) {
  std::ifstream input(file_name_.c_str(), std::ios::binary);
  if (!input) {
    EXIT("Couldn't load shader file " + file_name_ + "!");
  }

  // Read data into the buffer; needs to be char for istreambuf to work
  std::vector<char> buffer((
      std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));

  // Compile GLSL into SPIR-V
  const shaderc_compilation_result_t comp_results =
    shaderc_compile_into_spv(
        compiler,
        buffer.data(),
        SCAST_U32(buffer.size()),
        GetShadercShaderKind(),
        file_name_.c_str(),
        entry_point_.c_str(),
        shaderc_compile_options_t());

  shaderc_compilation_status comp_status =
    shaderc_result_get_compilation_status(comp_results);

  if (comp_status != shaderc_compilation_status_success) {
    eastl::string comp_err_msg = shaderc_result_get_error_message(comp_results);
    if (compiled_once_) {
      // Don't change shaders but report it
      ELOG_ERR("Reload of shader " << file_name_ << " failed:\n" <<
               comp_err_msg << "\n" << "Using initial shaders.");

      return current_stage_create_info_;
    }
    else {
      EXIT("Couldn't compile shader " << file_name_ << ":\n" <<
           comp_err_msg);
    }
  }


  VkShaderModuleCreateInfo module_create_info =
    tools::inits::ShaderModuleCreateInfo();
  module_create_info.codeSize = SCAST_U32(shaderc_result_get_length(
                                            comp_results));
  module_create_info.pCode =
    reinterpret_cast<const uint32_t *>(shaderc_result_get_bytes(comp_results));

  VkShaderModule module = VK_NULL_HANDLE;
  VK_CHECK_RESULT(vkCreateShaderModule(
      device.device(),
      &module_create_info,
      nullptr,
      &module));

  // Shutdown the module which has already been loaded in case the shader
  // is being recompiled to avoid memory leaks
  ShutdownModule(device);

  current_stage_create_info_ =
    tools::inits::PipelineShaderStageCreateInfo();
  current_stage_create_info_.stage = GetVkShaderType();
  current_stage_create_info_.module = module;
  current_stage_create_info_.pName = entry_point_.c_str();
  if (info_entries_.empty()) {
    current_stage_create_info_.pSpecializationInfo = nullptr; 
  }
  else {
    //spec_info_ = eastl::make_unique<VkSpecializationInfo>();
    spec_info_.pData = infos_data_.data();
    spec_info_.dataSize = SCAST_U32(infos_data_.size());
    spec_info_.mapEntryCount = SCAST_U32(info_entries_.size());
    spec_info_.pMapEntries = info_entries_.data();

    current_stage_create_info_.pSpecializationInfo = &spec_info_; 
  }

  compiled_once_ = true;

  return current_stage_create_info_;
}

const VkShaderStageFlagBits MaterialShader::GetVkShaderType() const {
  switch (type_) {
    case ShaderTypes::VERTEX: {
      return VK_SHADER_STAGE_VERTEX_BIT;
      break;
    }
    case ShaderTypes::FRAGMENT: {
      return VK_SHADER_STAGE_FRAGMENT_BIT;
      break;
    }
    default:{
      EXIT("This shader type is not supported!");
    }
  }
}

const shaderc_shader_kind MaterialShader::GetShadercShaderKind() const {
  switch (type_) {
    case ShaderTypes::VERTEX: {
      return shaderc_glsl_vertex_shader;
      break;
    }
    case ShaderTypes::FRAGMENT: {
      return shaderc_glsl_fragment_shader;
      break;
    }
    default:{
      EXIT("This shader type is not supported!");
    }
  }
}

Material::Material()
    : name_(),
      pipeline_(VK_NULL_HANDLE),
      modules_({VK_NULL_HANDLE}),
      builder_() {}

void Material::Init(const eastl::string &name) {
  name_ = name;

  LOG("Initialised Mat " + name_ + ".");
}

void Material::Shutdown(const VulkanDevice &device) {
  uint32_t modules_count = SCAST_U32(modules_.size());
  for (uint32_t i = 0U; i < modules_count; i++) {
    if (modules_[i] != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device.device(), modules_[i], nullptr);
      modules_[i] = VK_NULL_HANDLE;
    }
  }

  ShutdownPipeline(device);
  LOG("Shutdown material " << name_);
}

void Material::BindPipeline(VkCommandBuffer cmd_buff,
                            VkPipelineBindPoint bind_point) const {
  vkCmdBindPipeline(
      cmd_buff,
      bind_point,
      pipeline_);
}

MaterialBuilder::MaterialBuilder(
    const VertexSetup &vertex_setup,
    const eastl::string mat_name,
    VkPipelineLayout pipe_layout,
    VkRenderPass render_pass,
    VkFrontFace front_face,
    uint32_t subpass_idx,
    const szt::Viewport &viewport)
    : shaders_(),
      mat_name_(mat_name),
      vertex_size_(vertex_setup.vertex_size()),
      depth_test_enable_(VK_FALSE),
      depth_write_enable_(VK_FALSE),
      pipe_layout_(pipe_layout),
      front_face_(front_face),
      render_pass_(render_pass),
      subpass_idx_(subpass_idx),
      color_blend_state_create_info_(),
      vertex_setup_(),
      viewport_(viewport) {
  vertex_setup_ = eastl::make_unique<VertexSetup>(vertex_setup);
}

void MaterialBuilder::GetVertexInputBindingDescription(
    eastl::vector<VkVertexInputBindingDescription> &bindings) const {
  uint32_t layouts_count = vertex_setup_->num_elements();
  for (uint32_t i = 0U; i < layouts_count; i++) {
    VkVertexInputBindingDescription structure;
    structure.stride = vertex_setup_->GetElementSize(i);
    structure.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    structure.binding = i;

    bindings.push_back(structure);
  }
}

void MaterialBuilder::AddColorBlendStateCreateInfo(
    VkBool32                                      logic_op_enable,
    VkLogicOp                                     logic_op,
    float                                         blend_constants[4U]) {
  memcpy(blend_constants_.data(), blend_constants, sizeof(float) * 4U);

  color_blend_state_create_info_ =
    tools::inits::PipelineColorBlendStateCreateInfo(
      logic_op_enable,
      logic_op,
      SCAST_U32(color_blend_attachments_.size()),
      color_blend_attachments_.data(),
      blend_constants_.data());
}

void MaterialBuilder::AddColorBlendAttachment(
    VkBool32                 blend_enable,
    VkBlendFactor            src_color_blend_factor,
    VkBlendFactor            dst_color_blend_factor,
    VkBlendOp                color_blend_op,
    VkBlendFactor            src_alpha_blend_factor,
    VkBlendFactor            dst_alpha_blend_factor,
    VkBlendOp                alpha_blend_op,
    VkColorComponentFlags    color_write_mask) {
  color_blend_attachments_.push_back(
      tools::inits::PipelineColorBlendAttachmentState(
          blend_enable,
          src_color_blend_factor,
          dst_color_blend_factor,
          color_blend_op,
          src_alpha_blend_factor,
          dst_alpha_blend_factor,
          alpha_blend_op,
          color_write_mask));

  // Call it here so that if AddColorBlendStateCreateInfo had been called
  // before calling AddColorBlendAttachment the cached value can update
  // the number of color blend attachments.
  AddColorBlendStateCreateInfo(
      color_blend_state_create_info_.logicOpEnable,
      color_blend_state_create_info_.logicOp,
      blend_constants_.data());
}

void MaterialBuilder::GetVertexInputAttributeDescriptors(
    eastl::vector<VkVertexInputAttributeDescription> &attributes) const {
  uint32_t layouts_count = vertex_setup_->num_elements();
  for (uint32_t i = 0U; i < layouts_count; i++) {
    VkVertexInputAttributeDescription input_attribute_description;
    input_attribute_description.binding = i;
    input_attribute_description.location = i;
    input_attribute_description.offset = 0U;
    input_attribute_description.format =
      vertex_setup_->GetElementVulkanFormat(i);

    attributes.push_back(input_attribute_description);
  }
}

void MaterialBuilder::AddShader(eastl::unique_ptr<MaterialShader> shader) {
  shaders_.push_back(eastl::move(shader));
}

void Material::CacheBuilder(eastl::unique_ptr<MaterialBuilder> builder) {
  builder_ = eastl::move(builder);
}

void Material::CompileShaders(
    const VulkanDevice &device,
    eastl::vector<VkPipelineShaderStageCreateInfo> &stage_creates_out) {
  shaderc_compiler_t compiler = shaderc_compiler_initialize();

  uint32_t shader_stages_count = SCAST_U32(builder_->shaders().size());
  for (uint32_t i = 0U; i < shader_stages_count; i++) {
    VkPipelineShaderStageCreateInfo stage =
      builder_->shaders()[i]->Compile(device, compiler);
    stage_creates_out.push_back(stage);
    uint8_t idx = tools::ToUnderlying(builder_->shaders()[i]->type());
    modules_[idx] =
      stage_creates_out.back().module;
  }
}

void Material::CreatePipeline(
    const VulkanDevice &device,
    eastl::vector<VkPipelineShaderStageCreateInfo> &stage_create_infos) {
  // Setup the vertex input
  eastl::vector<VkVertexInputBindingDescription> bindings;
  eastl::vector<VkVertexInputAttributeDescription> attributes;
  builder_->GetVertexInputBindingDescription(bindings);
  builder_->GetVertexInputAttributeDescriptors(attributes);
 VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info =
    tools::inits::PipelineVertexInputStateCreateInfo();
  vertex_input_state_create_info.vertexBindingDescriptionCount =
    SCAST_U32(bindings.size());
  vertex_input_state_create_info.pVertexBindingDescriptions =
    bindings.data();
  
  vertex_input_state_create_info.vertexAttributeDescriptionCount =
    SCAST_U32(attributes.size());
  vertex_input_state_create_info.pVertexAttributeDescriptions =
    attributes.data();
  
  VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info =
    tools::inits::PipelineInputAssemblyStateCreateInfo();
  input_assembly_state_create_info.topology =
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly_state_create_info.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport;
  viewport.x = SCAST_FLOAT(builder_->viewport().x);
  viewport.y = SCAST_FLOAT(builder_->viewport().y);
  viewport.width = SCAST_FLOAT(builder_->viewport().width);
  viewport.height = SCAST_FLOAT(builder_->viewport().height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;

  VkRect2D scissor_area;
  scissor_area.offset.y = scissor_area.offset.x = 0U;
  scissor_area.extent.height = builder_->viewport().height;
  scissor_area.extent.width = builder_->viewport().width;

  VkPipelineViewportStateCreateInfo viewport_state_create_info =
    tools::inits::PipelineViewportStateCreateInfo();
  viewport_state_create_info.viewportCount = 1U;
  viewport_state_create_info.pViewports = &viewport;
  viewport_state_create_info.scissorCount = 1U;
  viewport_state_create_info.pScissors = &scissor_area;

  VkPipelineRasterizationStateCreateInfo raster_state_create_info =
    tools::inits::PipelineRasterizationStateCreateInfo();
  raster_state_create_info.depthClampEnable = VK_FALSE;
  raster_state_create_info.rasterizerDiscardEnable = VK_FALSE;
  raster_state_create_info.polygonMode = VK_POLYGON_MODE_FILL;
  raster_state_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
  raster_state_create_info.frontFace = builder_->front_face();
  raster_state_create_info.depthBiasEnable = VK_FALSE;
  raster_state_create_info.depthBiasConstantFactor = 0.f;
  raster_state_create_info.depthBiasClamp = 0.f;
  raster_state_create_info.depthBiasClamp = 0.f;
  raster_state_create_info.depthBiasSlopeFactor = 0.f;
  raster_state_create_info.lineWidth = 1.f;

  VkPipelineMultisampleStateCreateInfo multsampl_state_create_info =
    tools::inits::PipelineMultisampleStateCreateInfo();
  multsampl_state_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multsampl_state_create_info.sampleShadingEnable = VK_FALSE;
  multsampl_state_create_info.minSampleShading = 0.f;
  multsampl_state_create_info.pSampleMask = nullptr;
  multsampl_state_create_info.alphaToCoverageEnable = VK_FALSE;
  multsampl_state_create_info.alphaToOneEnable = VK_FALSE;

  VkPipelineDepthStencilStateCreateInfo depth_stenc_state_create_info =
    tools::inits::PipelineDepthStencilStateCreateInfo();
  depth_stenc_state_create_info.depthTestEnable = builder_->depth_test_enable();
  depth_stenc_state_create_info.depthWriteEnable =
    builder_->depth_write_enable();
  depth_stenc_state_create_info.depthCompareOp = VK_COMPARE_OP_LESS;
  depth_stenc_state_create_info.depthBoundsTestEnable = VK_FALSE;
  depth_stenc_state_create_info.stencilTestEnable = VK_FALSE;
  VkStencilOpState stencil_op_state;
  stencil_op_state.failOp = VK_STENCIL_OP_KEEP;
  stencil_op_state.passOp = VK_STENCIL_OP_KEEP;
  stencil_op_state.depthFailOp = VK_STENCIL_OP_KEEP;
  stencil_op_state.compareOp = VK_COMPARE_OP_NEVER;
  depth_stenc_state_create_info.front = stencil_op_state;
  depth_stenc_state_create_info.back = stencil_op_state;
  depth_stenc_state_create_info.minDepthBounds = 0.f;
  depth_stenc_state_create_info.maxDepthBounds = 0.f;

  VkGraphicsPipelineCreateInfo pipe_create_info =
    tools::inits::GraphicsPipelineCreateInfo();
  pipe_create_info.stageCount = SCAST_U32(stage_create_infos.size());
  pipe_create_info.pStages = stage_create_infos.data();
  pipe_create_info.pVertexInputState = &vertex_input_state_create_info;
  pipe_create_info.pInputAssemblyState = &input_assembly_state_create_info;
  pipe_create_info.pTessellationState = nullptr;
  pipe_create_info.pViewportState = &viewport_state_create_info;
  pipe_create_info.pRasterizationState = &raster_state_create_info;
  pipe_create_info.pMultisampleState = &multsampl_state_create_info;
  pipe_create_info.pDepthStencilState = &depth_stenc_state_create_info;
  VkPipelineColorBlendStateCreateInfo color_blend_state_create_info =
    builder_->color_blend_state_create_info();
  pipe_create_info.pColorBlendState = &color_blend_state_create_info;
  pipe_create_info.pDynamicState = nullptr;
  pipe_create_info.layout = builder_->pipe_layout();
  pipe_create_info.renderPass = builder_->render_pass();
  pipe_create_info.subpass = builder_->subpass_idx();
  pipe_create_info.basePipelineHandle = VK_NULL_HANDLE;
  pipe_create_info.basePipelineIndex = 0U;

  VK_CHECK_RESULT(vkCreateGraphicsPipelines(
      device.device(),
      VK_NULL_HANDLE,
      1U,
      &pipe_create_info,
      nullptr,
      &pipeline_));
}

void Material::InitPipeline(
    const VulkanDevice &device,
    eastl::unique_ptr<MaterialBuilder> builder) {
  CacheBuilder(eastl::move(builder));

  eastl::vector<VkPipelineShaderStageCreateInfo> module_create_builders;
  CompileShaders(device, module_create_builders);

  CreatePipeline(device, module_create_builders);

  LOG("Initialised pipe of Mat " + name_ + ".");
}

void Material::Reload(const VulkanDevice &device) {
  ShutdownPipeline(device);

  eastl::vector<VkPipelineShaderStageCreateInfo> module_create_builders;
  CompileShaders(device, module_create_builders);

  CreatePipeline(device, module_create_builders);

  LOG("Reloaded pipe and shaders of Mat " + name_ + ".");
}


void MaterialBuilder::SetDepthTestEnable(VkBool32 enable) {
  depth_test_enable_ = enable;
}

void MaterialBuilder::SetDepthWriteEnable(VkBool32 enable) {
  depth_write_enable_ = enable;
}

void Material::ShutdownPipeline(const VulkanDevice &device) {
  if (pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(device.device(), pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
  }
}

} // namespace vks
