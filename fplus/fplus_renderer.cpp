#include <fplus_renderer.h>
#include <base_system.h>
#include <vulkan_device.h>
#include <array>
#include <vulkan_tools.h>
#include <model.h>
#include <cassert>
#include <logger.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <material_constants.h>
#include <camera.h>
#include <glm/gtc/type_ptr.hpp>
#include <material_texture_type.h>
#include <vertex_setup.h>
#include <EASTL/vector.h>
#include <random>
#include <cstring>
#include <vulkan_texture.h>
#include <vulkan_image.h>
#include <meshes_heap_manager.h>

namespace vks {

extern const VkFormat kColourBufferFormat = VK_FORMAT_B8G8R8A8_SRGB;
const VkFormat kDiffuseAlbedoFormat = VK_FORMAT_R8G8B8A8_UNORM;
const VkFormat kSpecularAlbedoFormat = VK_FORMAT_R8G8B8A8_UNORM;
const VkFormat kNormalFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
const VkFormat kPositionFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
const VkFormat kSSAOFormat = VK_FORMAT_R8_UNORM;
const VkFormat kAccumulationFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
extern const int32_t kWindowWidth;
extern const int32_t kWindowHeight;
const uint32_t kTileSize = 16U;
const uint32_t kMaxLightsPerTile = 50U;
const uint32_t kMaxLightsScene = 3600U;
const uint32_t kWidthInTiles = kWindowWidth / kTileSize;
const uint32_t kHeightInTiles = kWindowHeight / kTileSize;
const uint32_t kTotalTilesNum = kWidthInTiles * kHeightInTiles;
const uint32_t kProjViewMatricesBindingPos = 0U;
const uint32_t kDepthBufferBindingPos = 1U;
const uint32_t kDiffuseTexturesArrayBindingPos = 2U;
const uint32_t kAmbientTexturesArrayBindingPos = 3U;
const uint32_t kSpecularTexturesArrayBindingPos = 4U;
const uint32_t kNormalTexturesArrayBindingPos = 5U;
const uint32_t kRoughnessTexturesArrayBindingPos = 6U;
const uint32_t kAccumulationBufferBindingPos = 7U;
const uint32_t kLightsArrayBindingPos = 8U;
const uint32_t kLightsIndicesBindingPos = 9U;
const uint32_t kMatConstsArrayBindingPos = 11U;
extern const uint32_t kModelMatxsBufferBindPos;
extern const uint32_t kMaterialIDsBufferBindPos;
const uint32_t kSpecInfoDrawCmdsCountID = 0U;
const uint32_t kUniformBufferDescCount = 5U;
const uint32_t kSetsCount = 3U;
const uint32_t kBindingsCount = 10U;
//const uint32_t kSSAOBuffersBindingPos = 8U;
//const uint32_t kSSAOKernelBindingPos = 9U;
const uint32_t kMaxNumUniformBuffers = 5U;
const uint32_t kMaxNumSSBOs = 30U;
const uint32_t kMaxNumMatInstances = 30U;
const uint32_t kNumMeshesSpecConstPos = 0U;
const uint32_t kNumMaterialsSpecConstPos = 0U;
//const uint32_t kSSAOKernelSizeSpecConstPos = 0U;
//const uint32_t kSSAONoiseTextureSizeSpecConstPos = 0U;
//const uint32_t kSSAORadiusSizeSpecConstPos = 1U;
//const uint32_t kNumIndirectDrawsSpecConstPos = 1U;
const uint32_t kNumLightsSpecConstPos = 1U;
const uint32_t kTonemapExposureSpecConstPos = 0U;
const float kTonemapExposure =0.8f;
//extern const uint32_t kVertexBuffersBaseBindPos;
//extern const uint32_t kIndirectDrawCmdsBindingPos;
//extern const uint32_t kIdxBufferBindPos;
const uint32_t kTileSizeSpecConstPos = 0U;
const uint32_t kMaxLightsPerTileSpecConstPos = 2U;
const uint32_t kRasterWidthSpecConstPos = 3U;
const uint32_t kRasterHeightSpecConstPos = 4U;
const eastl::string kBaseShaderAssetsPath = STR(ASSETS_FOLDER) "shaders/";

//const uint32_t kIndirectDrawCmdsBindingPos = 4U;
//const uint32_t kSSAOKernelSize = 16U;
//const float kSSAORadius = 2.f;
//const uint32_t kSSAONoiseTextureSize = 4U;

FPlusRenderer::FPlusRenderer()
  : depth_prepass_renderpass_(),
  shade_renderpass_(),
  framebuffers_(),
  depth_prepass_framebuffer_(),
  current_swapchain_img_(0U),
  cmd_buffers_(),
  cmd_buff_compute_(VK_NULL_HANDLE),
  cmd_buff_depth_prepass_(VK_NULL_HANDLE),
  depth_prepass_complete_semaphore_(VK_NULL_HANDLE),
  light_culling_complete_semaphore_(VK_NULL_HANDLE),
  accum_buffer_(),
  depth_buffer_(),
  depth_buffer_depth_view_(nullptr),
  depth_prepass_material_(nullptr),
  lights_cull_material_(nullptr),
  shading_material_(nullptr),
  tonemap_material_(nullptr),
  dummy_texture_(),
  //indirect_draw_cmds_(),
  //indirect_draw_buff_(),
  desc_set_layouts_(),
  desc_sets_(),
  desc_pool_(VK_NULL_HANDLE),
  pipe_layouts_(),
  main_static_buff_(),
  light_idxs_buff_(),
  proj_mat_(1.f),
  view_mat_(1.f),
  inv_proj_mat_(1.f),
  inv_view_mat_(1.f),
  cam_(nullptr),
  aniso_sampler_(VK_NULL_HANDLE),
  nearest_sampler_(VK_NULL_HANDLE),
  nearest_sampler_repeat_(VK_NULL_HANDLE),
  registered_models_(),
  fullscreenquad_(nullptr) {}

void FPlusRenderer::Init(szt::Camera *cam) {
  cam_ = cam;

  SetupSamplers(vulkan()->device());
  SetupDescriptorPool(vulkan()->device());
  
  model_manager()->set_shade_material_name("g_store");
  model_manager()->set_aniso_sampler(aniso_sampler_);
  model_manager()->set_sets_desc_pool(desc_pool_);

  texture_manager()->Load2DTexture(
     vulkan()->device(),
     STR(ASSETS_FOLDER) "dummy.ktx", 
     VK_FORMAT_BC2_UNORM_BLOCK,
     &dummy_texture_,
     aniso_sampler_);

  UpdatePVMatrices();
  SetupMaterials(vulkan()->device());
  SetupRenderPass(vulkan()->device());
  SetupFrameBuffers(vulkan()->device());
  CreateSemaphores(vulkan()->device());
  CreateCommandBuffers(vulkan()->device());
}

void FPlusRenderer::Shutdown() {
  vkDeviceWaitIdle(vulkan()->device().device());

  if (desc_pool_ != VK_NULL_HANDLE) {
    VK_CHECK_RESULT(vkResetDescriptorPool(
        vulkan()->device().device(),
        desc_pool_,
        0U));
    vkDestroyDescriptorPool(
      vulkan()->device().device(),
      desc_pool_,
      nullptr);

    desc_pool_ = VK_NULL_HANDLE;
  }

  if (aniso_sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(vulkan()->device().device(), aniso_sampler_, nullptr);
    aniso_sampler_ = VK_NULL_HANDLE;
  }
  if (nearest_sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(vulkan()->device().device(), nearest_sampler_, nullptr);
    nearest_sampler_ = VK_NULL_HANDLE;
  }
  if (nearest_sampler_repeat_ != VK_NULL_HANDLE) {
    vkDestroySampler(vulkan()->device().device(), nearest_sampler_repeat_, nullptr);
    nearest_sampler_repeat_ = VK_NULL_HANDLE;
  }


  for (uint32_t i = 0U; i < PipeLayoutTypes::num_items; i++) {
    vkDestroyPipelineLayout(
        vulkan()->device().device(),
        pipe_layouts_[i],
        nullptr);
  }

  for (uint32_t i = 0U; i < DescSetLayoutTypes::num_items; i++) {
    vkDestroyDescriptorSetLayout(
        vulkan()->device().device(),
        desc_set_layouts_[i],
        nullptr);
  }

  if (depth_prepass_complete_semaphore_!= VK_NULL_HANDLE) {
    vkDestroySemaphore(vulkan()->device().device(), depth_prepass_complete_semaphore_,
                       nullptr);
    depth_prepass_complete_semaphore_ = VK_NULL_HANDLE;
  }
  if (light_culling_complete_semaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(vulkan()->device().device(), light_culling_complete_semaphore_,
                       nullptr);
    light_culling_complete_semaphore_ = VK_NULL_HANDLE;
  }


  light_idxs_buff_.Shutdown(vulkan()->device());
  main_static_buff_.Shutdown(vulkan()->device());
  framebuffers_.clear();
  depth_prepass_framebuffer_.reset(nullptr);
  shade_renderpass_.reset(nullptr);
  depth_prepass_renderpass_.reset(nullptr);
}

void FPlusRenderer::PreRender() {
  UpdateBuffers(vulkan()->device());

  vulkan()->swapchain().AcquireNextImage(
      vulkan()->device(),
      vulkan()->image_available_semaphore(),
      current_swapchain_img_);
}

void FPlusRenderer::UpdateBuffers(const VulkanDevice &device) {
  UpdatePVMatrices();
  eastl::vector<Light> transformed_lights;
  UpdateLights(transformed_lights);

  // Cache some sizes
  uint32_t num_mat_instances = material_manager()->GetMaterialInstancesCount();
  uint32_t num_lights = SCAST_U32(transformed_lights.size());
  uint32_t mat4_size = SCAST_U32(sizeof(glm::mat4));
  uint32_t mat4_group_size = mat4_size * 4U;
  uint32_t lights_array_size = (SCAST_U32(sizeof(Light)) * num_lights);
  uint32_t mat_consts_array_size =
    (SCAST_U32(sizeof(MaterialConstants)) * num_mat_instances);

  // Upload data to the buffers
  eastl::array<glm::mat4, 4U> matxs_initial_data = {
    proj_mat_, view_mat_ , inv_proj_mat_, inv_view_mat_};

  void *mapped = nullptr;
  main_static_buff_.Map(device, &mapped);
  uint8_t * mapped_u8 = static_cast<uint8_t *>(mapped);

  memcpy(mapped, matxs_initial_data.data(), mat4_group_size);
  mapped_u8 += mat4_group_size;

  memcpy(mapped_u8, transformed_lights.data(), lights_array_size);
  mapped_u8 += lights_array_size;

  memcpy(mapped_u8, mat_consts_.data(), mat_consts_array_size);

  main_static_buff_.Unmap(device);
}

void FPlusRenderer::Render() {
  eastl::array<VkSemaphore, 2U> wait_semaphores = {
    vulkan()->image_available_semaphore(),
    light_culling_complete_semaphore_
  };
  VkSemaphore signal_semaphore = vulkan()->rendering_finished_semaphore();
  eastl::array<VkPipelineStageFlags, 2U> wait_stages{ VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
  };
  VkCommandBuffer cmd_buff =
    cmd_buffers_[current_swapchain_img_];
  VkSubmitInfo submit_info = tools::inits::SubmitInfo();
  submit_info.waitSemaphoreCount = wait_semaphores.size();
  submit_info.pWaitSemaphores = wait_semaphores.data();
  submit_info.pWaitDstStageMask = wait_stages.data();
  submit_info.commandBufferCount = 1U;
  submit_info.pCommandBuffers = &cmd_buff;
  submit_info.signalSemaphoreCount = 1U;
  submit_info.pSignalSemaphores = &signal_semaphore;

  VkSubmitInfo submit_info_depth_prepass = tools::inits::SubmitInfo();
  submit_info_depth_prepass.waitSemaphoreCount = 0U;
  submit_info_depth_prepass.pWaitSemaphores = nullptr;
  submit_info_depth_prepass.pWaitDstStageMask = nullptr;
  submit_info_depth_prepass.commandBufferCount = 1U;
  submit_info_depth_prepass.pCommandBuffers = &cmd_buff_depth_prepass_;
  submit_info_depth_prepass.signalSemaphoreCount = 1U;
  submit_info_depth_prepass.pSignalSemaphores = &depth_prepass_complete_semaphore_;

  VkPipelineStageFlags cull_wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkSubmitInfo submit_info_cull = tools::inits::SubmitInfo();
  submit_info_cull.waitSemaphoreCount = 1U;
  submit_info_cull.pWaitSemaphores = &depth_prepass_complete_semaphore_;
  submit_info_cull.pWaitDstStageMask = &cull_wait_stage;
  submit_info_cull.commandBufferCount = 1U;
  submit_info_cull.pCommandBuffers = &cmd_buff_compute_;
  submit_info_cull.signalSemaphoreCount = 1U;
  submit_info_cull.pSignalSemaphores = &light_culling_complete_semaphore_;

  eastl::array<VkSubmitInfo, 1U> queue_graphics_depth_infos = {
    submit_info_depth_prepass,
  };
  eastl::array<VkSubmitInfo, 1U> queue_compute_infos = {
    submit_info_cull
  };
  eastl::array<VkSubmitInfo, 1U> queue_graphics_infos = {
    submit_info
  };

  VK_CHECK_RESULT(vkQueueSubmit(
      vulkan()->device().graphics_queue().queue,
      queue_graphics_depth_infos.size(),
      queue_graphics_depth_infos.data(),
      VK_NULL_HANDLE));
  VK_CHECK_RESULT(vkQueueSubmit(
      vulkan()->device().compute_queue().queue,
      queue_compute_infos.size(),
      queue_compute_infos.data(),
      VK_NULL_HANDLE));
  VK_CHECK_RESULT(vkQueueSubmit(
      vulkan()->device().graphics_queue().queue,
      queue_graphics_infos.size(),
      queue_graphics_infos.data(),
      VK_NULL_HANDLE));
}

void FPlusRenderer::PostRender() {
  vulkan()->swapchain().Present(
      vulkan()->device().present_queue(),
      vulkan()->rendering_finished_semaphore());
}

void FPlusRenderer::SetupRenderPass(const VulkanDevice &device) {
  depth_prepass_renderpass_ = eastl::make_unique<Renderpass>("depth_prepass");
  shade_renderpass_ = eastl::make_unique<Renderpass>("shade_pass");

  // Colour buffer target 
  uint32_t col_buf_id = shade_renderpass_->AddAttachment(
      0U,
      vulkan()->swapchain().GetSurfaceFormat(),
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR); 

  // Depth buffer target
  uint32_t depth_buf_shade_id = shade_renderpass_->AddAttachment(
      0U,
      device.depth_format(),
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  uint32_t depth_buf_prepass_id = depth_prepass_renderpass_->AddAttachment(
      0U,
      device.depth_format(),
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

  // Accumulation buffer
  uint32_t accum_id = shade_renderpass_->AddAttachment(
      0U,
      kAccumulationFormat,
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); 
  
  //// SSAO buffer
  //uint32_t ssao_id = renderpass_->AddAttachment(
  //    0U,
  //    kSSAOFormat,
  //    VK_SAMPLE_COUNT_1_BIT,
  //    VK_ATTACHMENT_LOAD_OP_CLEAR,
  //    VK_ATTACHMENT_STORE_OP_STORE,
  //    VK_ATTACHMENT_LOAD_OP_DONT_CARE,
  //    VK_ATTACHMENT_STORE_OP_DONT_CARE,
  //    VK_IMAGE_LAYOUT_UNDEFINED,
  //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); 
  //
  //// SSAO blurred buffer
  //uint32_t ssao_blur_id = renderpass_->AddAttachment(
  //    0U,
  //    kSSAOFormat,
  //    VK_SAMPLE_COUNT_1_BIT,
  //    VK_ATTACHMENT_LOAD_OP_CLEAR,
  //    VK_ATTACHMENT_STORE_OP_STORE,
  //    VK_ATTACHMENT_LOAD_OP_DONT_CARE,
  //    VK_ATTACHMENT_STORE_OP_DONT_CARE,
  //    VK_IMAGE_LAYOUT_UNDEFINED,
  //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); 

  uint32_t first_sub_prepass_id = depth_prepass_renderpass_->AddSubpass(
      "depth_prepass",
      VK_PIPELINE_BIND_POINT_GRAPHICS);
  // Depth
  depth_prepass_renderpass_->AddSubpassDepthAttachmentRef(
      first_sub_prepass_id,
      depth_buf_prepass_id,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  
  uint32_t first_sub_shade_id = shade_renderpass_->AddSubpass(
      "shade",
      VK_PIPELINE_BIND_POINT_GRAPHICS);
  // Depth
  shade_renderpass_->AddSubpassDepthAttachmentRef(
      first_sub_shade_id,
      depth_buf_shade_id,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  // Accumulation
  shade_renderpass_->AddSubpassColourAttachmentRef(
      first_sub_shade_id,
      accum_id,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  
  //uint32_t ssao_sub_id = renderpass_->AddSubpass(
  //    "ssao",
  //    VK_PIPELINE_BIND_POINT_GRAPHICS);
  //renderpass_->AddSubpassColourAttachmentRef(
  //    ssao_sub_id,
  //    ssao_id,
  //    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  //renderpass_->AddSubpassInputAttachmentRef(
  //    ssao_sub_id,
  //    depth_buf_id,
  //    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
  //renderpass_->AddSubpassInputAttachmentRef(
  //    ssao_sub_id,
  //    norm_id,
  //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  //renderpass_->AddSubpassPreserveAttachmentRef(
  //    ssao_sub_id,
  //    diff_albedo_id);
  //renderpass_->AddSubpassPreserveAttachmentRef(
  //    ssao_sub_id,
  //    spec_albedo_id);
  //renderpass_->AddSubpassPreserveAttachmentRef(
  //    ssao_sub_id,
  //    norm_id);
  //renderpass_->AddSubpassPreserveAttachmentRef(
  //    ssao_sub_id,
  //    depth_buf_id);
  
  //uint32_t ssao_blur_sub_id = renderpass_->AddSubpass(
  //    "ssao_blur",
  //    VK_PIPELINE_BIND_POINT_GRAPHICS);
  //renderpass_->AddSubpassColourAttachmentRef(
  //    ssao_blur_sub_id,
  //    ssao_blur_id,
  //    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  //renderpass_->AddSubpassInputAttachmentRef(
  //    ssao_blur_sub_id,
  //    ssao_id,
  //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  //renderpass_->AddSubpassPreserveAttachmentRef(
  //    ssao_blur_sub_id,
  //    diff_albedo_id);
  //renderpass_->AddSubpassPreserveAttachmentRef(
  //    ssao_blur_sub_id,
  //    spec_albedo_id);
  //renderpass_->AddSubpassPreserveAttachmentRef(
  //    ssao_blur_sub_id,
  //    norm_id);
  //renderpass_->AddSubpassPreserveAttachmentRef(
  //    ssao_blur_sub_id,
  //    depth_buf_id);

  //renderpass_->AddSubpassInputAttachmentRef(
  //    lighting_sub_id,
  //    ssao_blur_id,
  //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  //renderpass_->AddSubpassInputAttachmentRef(
  //    lighting_sub_id,
  //    norm_id,
  //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  //renderpass_->AddSubpassInputAttachmentRef(
  //    lighting_sub_id,
  //    diff_albedo_id,
  //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  //renderpass_->AddSubpassInputAttachmentRef(
  //    lighting_sub_id,
  //    spec_albedo_id,
  //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  //renderpass_->AddSubpassInputAttachmentRef(
  //    lighting_sub_id,
  //    depth_buf_id,
  //    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
  //renderpass_->AddSubpassInputAttachmentRef(
  //    shading_sub_id,
  //    ssao_blur_id,
  //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  uint32_t third_sub_id = shade_renderpass_->AddSubpass(
      "tonemapping",
      VK_PIPELINE_BIND_POINT_GRAPHICS);
  shade_renderpass_->AddSubpassColourAttachmentRef(
      third_sub_id,
      col_buf_id,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  shade_renderpass_->AddSubpassInputAttachmentRef(
      third_sub_id,
      accum_id,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // Dependencies
  // Present to colour buffer, which is the last subpass
  shade_renderpass_->AddSubpassDependency(
      VK_SUBPASS_EXTERNAL,
      third_sub_id,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_MEMORY_READ_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_DEPENDENCY_BY_REGION_BIT);
  
  shade_renderpass_->AddSubpassDependency(
      first_sub_shade_id,
      third_sub_id,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT,
      VK_DEPENDENCY_BY_REGION_BIT);

  // Final subpass to present 
  shade_renderpass_->AddSubpassDependency(
      third_sub_id,
      VK_SUBPASS_EXTERNAL,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_ACCESS_MEMORY_READ_BIT,
      VK_DEPENDENCY_BY_REGION_BIT);

  //renderpass_->AddSubpassDependency(
  //    first_sub_id,
  //    ssao_sub_id,
  //    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
  //    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
  //    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  //    VK_ACCESS_SHADER_READ_BIT,
  //    VK_DEPENDENCY_BY_REGION_BIT);
  //
  //renderpass_->AddSubpassDependency(
  //    ssao_sub_id,
  //    ssao_blur_sub_id,
  //    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
  //    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
  //    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  //    VK_ACCESS_SHADER_READ_BIT,
  //    VK_DEPENDENCY_BY_REGION_BIT);
  //
  //renderpass_->AddSubpassDependency(
  //    ssao_blur_sub_id,
  //    lighting_sub_id,
  //    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
  //    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
  //    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  //    VK_ACCESS_SHADER_READ_BIT,
  //    VK_DEPENDENCY_BY_REGION_BIT);

  shade_renderpass_->CreateVulkanRenderpass(device);
  depth_prepass_renderpass_->CreateVulkanRenderpass(device);
}

void FPlusRenderer::CreateSemaphores(const VulkanDevice &device) {
  VkSemaphoreCreateInfo semaphore_create_info = {
    VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    nullptr,
    0U
  };

  VK_CHECK_RESULT(vkCreateSemaphore(device.device(), &semaphore_create_info,
                                    nullptr, &depth_prepass_complete_semaphore_));
  VK_CHECK_RESULT(vkCreateSemaphore(device.device(), &semaphore_create_info,
                                    nullptr, &light_culling_complete_semaphore_));
}

void FPlusRenderer::SetupFrameBuffers(const VulkanDevice &device) {
  // Accumulation buffer
  CreateFramebufferAttachment(
      device,
      kAccumulationFormat,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
      "accumulation",
      &accum_buffer_);
  
  // SSAO buffer
  //CreateFramebufferAttachment(
  //    device,
  //    kSSAOFormat,
  //    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
  //      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
  //    "ssao",
  //    &ssao_buffer_);
  
  // SSAO blur buffer
  //CreateFramebufferAttachment(
  //    device,
  //    kSSAOFormat,
  //    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
  //      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
  //    "ssao_blur",
  //    &ssao_blur_buffer_);

  // Depth buffer 
  CreateFramebufferAttachment(
      device,
      device.depth_format(),
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
      "depth",
      &depth_buffer_);

  VkImageViewCreateInfo depth_view_create_info =
    tools::inits::ImageViewCreateInfo(
      depth_buffer_->image()->image(),
      VK_IMAGE_VIEW_TYPE_2D,
      depth_buffer_->image()->format(),
      {
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY
      },
      {
        VK_IMAGE_ASPECT_DEPTH_BIT,
        0U,
        depth_buffer_->image()->mip_levels(),
        0U,
        1U
      });
  depth_buffer_depth_view_ = depth_buffer_->image()->CreateAdditionalImageView(
      device,
      depth_view_create_info);

  // Create the depth prepass framebuffer, which can be only one since it
  // isn't in the same renderpass as the swapchain images
  depth_prepass_framebuffer_ = eastl::make_unique<Framebuffer>(
        "depth_prepass",
        cam_->viewport().width,
        cam_->viewport().height,
        1U,
        depth_prepass_renderpass_.get());

  depth_prepass_framebuffer_->AddAttachment(depth_buffer_);
  depth_prepass_framebuffer_->CreateVulkanFramebuffer(device);

  const uint32_t num_swapchain_images = vulkan()->swapchain().GetNumImages();
  for (uint32_t i = 0U; i < num_swapchain_images; i++) {
    eastl::string name;
    name.sprintf("from_swapchain_%d",i);
    eastl::unique_ptr<Framebuffer> frmbuff = eastl::make_unique<Framebuffer>(
        name,
        cam_->viewport().width,
        cam_->viewport().height,
        1U,
        shade_renderpass_.get());

    frmbuff->AddAttachment(vulkan()->swapchain().images()[i]);
    frmbuff->AddAttachment(depth_buffer_);

    frmbuff->AddAttachment(accum_buffer_);
    
    //frmbuff->AddAttachment(ssao_buffer_);
    
    //frmbuff->AddAttachment(ssao_blur_buffer_);

    frmbuff->CreateVulkanFramebuffer(device);

    framebuffers_.push_back(eastl::move(frmbuff));
  }
}

void FPlusRenderer::CreateFramebufferAttachment(
    const VulkanDevice &device,
    VkFormat format,
    VkImageUsageFlags img_usage_flags,
    const eastl::string &name,
    VulkanTexture **attachment) const {
  texture_manager()->Create2DTextureFromData(
      device,
      name,
      nullptr,
      0U,
      cam_->viewport().width,
      cam_->viewport().height,
      format,
      attachment,
      VK_NULL_HANDLE,
      img_usage_flags);
}

void FPlusRenderer::RegisterModel(Model &model,
                                     const VertexSetup &g_store_vertex_setup) {
  registered_models_.push_back(&model);

  SetupDescriptorSetAndPipeLayout(vulkan()->device());
  model.CreateAndWriteDescriptorSets(vulkan()->device(),
      desc_set_layouts_[DescSetLayoutTypes::MODELS]);
  SetupUniformBuffers(vulkan()->device());
  SetupMaterialPipelines(vulkan()->device(), g_store_vertex_setup);
  SetupDescriptorSets(vulkan()->device());
  SetupFullscreenQuad(vulkan()->device());
  SetupGraphicsCommandBuffers(vulkan()->device());
  SetupComputeCommandBuffers(vulkan()->device());
 
  LOG("Registered model in FPlusRenderer.");
}

void FPlusRenderer::SetupMaterials(const VulkanDevice &device) {
  material_manager()->RegisterMaterialName("depth_prepass");
  material_manager()->RegisterMaterialName("lights_culling");
  material_manager()->RegisterMaterialName("shade");
}

void FPlusRenderer::SetupUniformBuffers(const VulkanDevice &device) {
  // Materials
  mat_consts_ = material_manager()->GetMaterialConstants();
  uint32_t num_mat_instances = material_manager()->GetMaterialInstancesCount();

  // Lights array
  eastl::vector<Light> transformed_lights;
  UpdateLights(transformed_lights);
  uint32_t num_lights = SCAST_U32(transformed_lights.size());

  // Cache some sizes
  uint32_t mat4_size = SCAST_U32(sizeof(glm::mat4));
  uint32_t mat4_group_size = mat4_size * 4U;
  uint32_t lights_array_size = (SCAST_U32(sizeof(Light)) * num_lights);
  uint32_t mat_consts_array_size =
    (SCAST_U32(sizeof(MaterialConstants)) * num_mat_instances);
  uint32_t lights_indices_array_size = SCAST_U32(sizeof(uint32_t)) * kMaxLightsPerTile * num_lights * kTotalTilesNum;
  uint32_t latest_size_offset = 0U;
  
  // Setup the random generation classes for the SSAO step
  //std::uniform_real_distribution<float> rnd_dist(0.f, 1.f);
  //std::uniform_real_distribution<float> rnd_dist_negative(-1.f, 1.f);
  //std::default_random_engine rnd_gen;

  //// Create the SSAO Sample kernel
  //eastl::vector<glm::vec3> ssao_kernel(kSSAOKernelSize);
  //for (uint32_t i = 0; i < kSSAOKernelSize; i++) {
  //  // Orient around Z-axis
  //  glm::vec3 sample(
  //      rnd_dist_negative(rnd_gen),
  //      rnd_dist_negative(rnd_gen),
  //      rnd_dist(rnd_gen));
  //  sample = glm::normalize(sample);

  //  // Scale sample positions within the hemisphere
  //  sample *= rnd_dist(rnd_gen);

  //  // Have distance from the origin to falloff as we generate more points
  //  float scale = SCAST_FLOAT(i) / SCAST_FLOAT(kSSAOKernelSize);
  //  scale = tools::Lerp(0.1f, 1.f, scale * scale);
  //  ssao_kernel[i] = glm::vec3(sample * scale);
  //}

  // Random noise for sampling the SSAO kernel and introducing high-frequency noise
  //eastl::vector<glm::vec2> ssao_noise(kSSAONoiseTextureSize * kSSAONoiseTextureSize);
  //uint32_t ssao_noise_count = kSSAONoiseTextureSize * kSSAONoiseTextureSize;
  //for (uint32_t i = 0; i < ssao_noise_count; i++) {
  //  ssao_noise[i] = glm::vec2(
  //      rnd_dist_negative(rnd_gen),
  //      rnd_dist_negative(rnd_gen));
  //}

  //// UV repetition scale for the noise texture
  //glm::vec2 uv_noise_scale = glm::vec2(
  //    SCAST_FLOAT(cam_->viewport().width) / SCAST_FLOAT(kSSAONoiseTextureSize),
  //    SCAST_FLOAT(cam_->viewport().height) / SCAST_FLOAT(kSSAONoiseTextureSize));

  // Main static buffer
  VulkanBufferInitInfo buff_init_info;
  buff_init_info.size = mat4_group_size +
    lights_array_size +
    mat_consts_array_size;
    //noise_uv_scale_size +
    //ssao_kernel_size;
  buff_init_info.memory_property_flags =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  buff_init_info.buffer_usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  main_static_buff_.Init(device, buff_init_info);

  // Upload data to it
  eastl::array<glm::mat4, 4U> matxs_initial_data = {
    proj_mat_, view_mat_ , inv_proj_mat_, inv_view_mat_};

  void *mapped = nullptr;
  main_static_buff_.Map(device, &mapped);
  uint8_t *mapped_u8 = static_cast<uint8_t *>(mapped);

  memcpy(mapped, matxs_initial_data.data(), mat4_group_size);
  mapped_u8 += mat4_group_size;

  memcpy(mapped_u8, transformed_lights.data(), lights_array_size);
  mapped_u8 += lights_array_size;
  
  memcpy(mapped_u8, mat_consts_.data(), mat_consts_array_size);
  mapped_u8 += mat_consts_array_size;

  //memcpy(mapped_u8, glm::value_ptr(uv_noise_scale), noise_uv_scale_size);
  //mapped_u8 += noise_uv_scale_size;
  //
  //memcpy(mapped_u8, ssao_kernel.data(), ssao_kernel_size);

  main_static_buff_.Unmap(device);

  // Lights ID buffer
  buff_init_info.size =
    lights_indices_array_size;
    //noise_uv_scale_size +
    //ssao_kernel_size;
  buff_init_info.memory_property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  buff_init_info.buffer_usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  light_idxs_buff_.Init(device, buff_init_info);


  // Upload as texture, even though we are in the buffers setup function;
  // this is because it can be wrapped around when sampling across the
  // whole colour target
  //VulkanTexture *noise_texture = nullptr;
  //texture_manager()->Create2DTextureFromData(
  //    device,
  //    "ssao_random_noise",
  //    ssao_noise.data(),
  //    SCAST_U32(ssao_noise.size() * sizeof(glm::vec2)),
  //    kSSAONoiseTextureSize,
  //    kSSAONoiseTextureSize,
  //    VK_FORMAT_R32G32_SFLOAT,
  //    &noise_texture,
  //    nearest_sampler_repeat_);
}

void FPlusRenderer::SetupDescriptorPool(const VulkanDevice &device) {
  eastl::vector<VkDescriptorPoolSize> pool_sizes;

  // Uniforms
  pool_sizes.push_back(tools::inits::DescriptorPoolSize(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      kMaxNumUniformBuffers));

  // Framebuffers
  pool_sizes.push_back(tools::inits::DescriptorPoolSize(
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      kMaxNumMatInstances * 
        SCAST_U32(MatTextureType::size) + 10U));

  // Input attachments
  pool_sizes.push_back(tools::inits::DescriptorPoolSize(
    VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
    10U));

  // Storage buffers
  pool_sizes.push_back(tools::inits::DescriptorPoolSize(
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      kMaxNumSSBOs));

  VkDescriptorPoolCreateInfo pool_create_info =
    tools::inits::DescriptrorPoolCreateInfo(
      DescSetLayoutTypes::num_items,
      SCAST_U32(pool_sizes.size()),
      pool_sizes.data());

  VK_CHECK_RESULT(vkCreateDescriptorPool(device.device(), &pool_create_info,
                  nullptr, &desc_pool_));
}

void FPlusRenderer::SetupDescriptorSetAndPipeLayout(
    const VulkanDevice &device) {
  eastl::vector<std::vector<VkDescriptorSetLayoutBinding>> bindings(
      DescSetLayoutTypes::num_items);

  // VP matrices
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kProjViewMatricesBindingPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
        VK_SHADER_STAGE_COMPUTE_BIT,
      nullptr));
  
  // Lights array
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kLightsArrayBindingPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
        VK_SHADER_STAGE_COMPUTE_BIT,
      nullptr));

  // Lights indirection indices
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kLightsIndicesBindingPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
      nullptr));
  
  // Material constants array
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kMatConstsArrayBindingPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  // Model matrices for all meshes
  bindings[DescSetLayoutTypes::MODELS].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kModelMatxsBufferBindPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  // Material IDs
  bindings[DescSetLayoutTypes::MODELS].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kMaterialIDsBufferBindPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  // Depth buffer
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kDepthBufferBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      1U,
      VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
      nullptr));

  uint32_t num_mat_instances = material_manager()->GetMaterialInstancesCount();
  // Diffuse textures as combined image samplers
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kDiffuseTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  // Ambient textures as combined image samplers 
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kAmbientTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  // Specular textures as combined image samplers 
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kSpecularTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  // Normal textures as combined image samplers 
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kNormalTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  // Roughness textures as combined image samplers 
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kRoughnessTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  // Accumulation buffer
  bindings[DescSetLayoutTypes::GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kAccumulationBufferBindingPos,
      VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
      1U,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  
  //// SSAO buffers (ssao target + ssao noise texture + ssao blurred target)
  //bindings[DescSetLayoutTypes::GENERIC].push_back(
  //  tools::inits::DescriptorSetLayoutBinding(
  //    kSSAOBuffersBindingPos,
  //    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
  //    3U,
  //    VK_SHADER_STAGE_FRAGMENT_BIT,
  //    nullptr));
  //
  //// SSAO kernel
  //bindings[DescSetLayoutTypes::GENERIC].push_back(
  //  tools::inits::DescriptorSetLayoutBinding(
  //    kSSAOKernelBindingPos,
  //    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
  //    1U,
  //    VK_SHADER_STAGE_FRAGMENT_BIT,
  //    nullptr));

  //// G-Buffers
  //for (uint32_t i = 0U; i < GBtypes::num_items; i++) {
  //  bindings[DescSetLayoutTypes::GENERIC].push_back(
  //    tools::inits::DescriptorSetLayoutBinding(
  //      kGBufferBaseBindingPos + i,
  //      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
  //      1U,
  //      VK_SHADER_STAGE_FRAGMENT_BIT,
  //      nullptr));
  //}


  for (uint32_t i = 0U; i < DescSetLayoutTypes::num_items; i++) {
    VkDescriptorSetLayoutCreateInfo set_layout_create_info =
      tools::inits::DescriptrorSetLayoutCreateInfo();
    set_layout_create_info.bindingCount =
      SCAST_U32(bindings[i].size());
    set_layout_create_info.pBindings =
      bindings[i].data();

    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device.device(),
        &set_layout_create_info,
        nullptr,
        &desc_set_layouts_[i]));

    //LOG("Desc set layout: " << desc_set_layouts_[i] << " b count: " << 
    //    set_layout_create_info.bindingCount);
  }
  
  // Allocate only local layouts
  eastl::array<VkDescriptorSetLayout, SetTypes::num_items>
  local_layouts = {
    desc_set_layouts_[DescSetLayoutTypes::GENERIC],
  };
  VkDescriptorSetAllocateInfo set_allocate_info =
    tools::inits::DescriptorSetAllocateInfo(
      desc_pool_,
      SCAST_U32(local_layouts.size()),
      local_layouts.data());

  VK_CHECK_RESULT(vkAllocateDescriptorSets(
        device.device(),
        &set_allocate_info,
        desc_sets_.data()));

  // Create pipeline layouts
  // Push constant for the meshes ID
  VkPushConstantRange push_const_range = {
    VK_SHADER_STAGE_VERTEX_BIT,
    0U,
    SCAST_U32(sizeof(uint32_t))
  };

  VkPipelineLayoutCreateInfo pipe_layout_create_info = tools::inits::PipelineLayoutCreateInfo(
      DescSetLayoutTypes::num_items,
      desc_set_layouts_.data(),
      1U,
      &push_const_range);

  VK_CHECK_RESULT(vkCreatePipelineLayout(
      device.device(),
      &pipe_layout_create_info,
      nullptr,
      &pipe_layouts_[PipeLayoutTypes::GENERIC]));
}

void FPlusRenderer::SetupDescriptorSets(const VulkanDevice &device) {
  // Update the descriptor set
  eastl::vector<VkWriteDescriptorSet> write_desc_sets;

  // Cache some sizes
  uint32_t num_mat_instances = material_manager()->GetMaterialInstancesCount();
  uint32_t num_lights = lights_manager()->GetNumLights();
  uint32_t mat4_size = SCAST_U32(sizeof(glm::mat4));
  uint32_t mat4_group_size = mat4_size * 4U;
  uint32_t lights_array_size = (SCAST_U32(sizeof(Light)) * num_lights);
  uint32_t mat_consts_array_size =
    (SCAST_U32(sizeof(MaterialConstants)) * num_mat_instances);
  uint32_t lights_indices_array_size = SCAST_U32(sizeof(uint32_t)) * kMaxLightsPerTile * num_lights * kTotalTilesNum;
  uint32_t lights_grid_size =
    SCAST_U32(sizeof(uint32_t)) *
    (kWindowWidth / kTileSize) * (kWindowHeight / kTileSize);
  uint32_t latest_size_offset = 0U;
  //uint32_t noise_uv_scale_size = SCAST_U32(sizeof(glm::vec2));
  //uint32_t ssao_kernel_size = SCAST_U32(sizeof(glm::vec3)) * kSSAOKernelSize;

  // VP matrices
  VkDescriptorBufferInfo desc_main_static_buff_info =
    main_static_buff_.GetDescriptorBufferInfo(mat4_group_size);
  latest_size_offset += mat4_group_size;
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kProjViewMatricesBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      nullptr,
      &desc_main_static_buff_info,
      nullptr));

  // Lights array
  VkDescriptorBufferInfo desc_lights_array_info =
    main_static_buff_.GetDescriptorBufferInfo(lights_array_size,
       latest_size_offset);
  latest_size_offset += lights_array_size;
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kLightsArrayBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      nullptr,
      &desc_lights_array_info,
      nullptr));
  
  // Lights indirection indices
  VkDescriptorBufferInfo desc_lights_indices_info =
    light_idxs_buff_.GetDescriptorBufferInfo(lights_indices_array_size);
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kLightsIndicesBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      nullptr,
      &desc_lights_indices_info,
      nullptr));

  // Material constants array
  VkDescriptorBufferInfo desc_mat_consts_info =
    main_static_buff_.GetDescriptorBufferInfo(mat_consts_array_size,
                                              latest_size_offset);
  latest_size_offset += mat_consts_array_size;
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kMatConstsArrayBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      nullptr,
      &desc_mat_consts_info,
      nullptr));
  
  
  //// SSAO kernel array
  //VkDescriptorBufferInfo desc_ssao_kernel_info =
  //  main_static_buff_.GetDescriptorBufferInfo(
  //      ssao_kernel_size + noise_uv_scale_size,
  //      mat4_group_size +
  //        lights_array_size +
  //        mat_consts_array_size);
  //write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
  //    desc_sets_[SetTypes::GENERIC],
  //    kSSAOKernelBindingPos,
  //    0U,
  //    1U,
  //    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
  //    nullptr,
  //    &desc_ssao_kernel_info,
  //    nullptr));

  // Depth buffer
  VkDescriptorImageInfo depth_buff_img_info =
    depth_buffer_->image()->GetDescriptorImageInfo(nearest_sampler_);
  depth_buff_img_info.imageView = *depth_buffer_depth_view_;
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kDepthBufferBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      &depth_buff_img_info,
      nullptr,
      nullptr));

  eastl::vector<VkDescriptorImageInfo> diff_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::DIFFUSE,
      diff_descs_image_infos);

  // Diffuse textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kDiffuseTexturesArrayBindingPos,
      0U,
      SCAST_U32(diff_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      diff_descs_image_infos.data(),
      nullptr,
      nullptr));
  
  eastl::vector<VkDescriptorImageInfo> amb_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::AMBIENT,
      amb_descs_image_infos);

  // Ambient textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kAmbientTexturesArrayBindingPos,
      0U,
      SCAST_U32(amb_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      amb_descs_image_infos.data(),
      nullptr,
      nullptr));
  
  eastl::vector<VkDescriptorImageInfo> spec_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::SPECULAR,
      spec_descs_image_infos);

  // Specular textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kSpecularTexturesArrayBindingPos,
      0U,
      SCAST_U32(spec_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      spec_descs_image_infos.data(),
      nullptr,
      nullptr));
  
  eastl::vector<VkDescriptorImageInfo> rough_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::SPECULAR_HIGHLIGHT,
      rough_descs_image_infos);

  // Roughness textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kRoughnessTexturesArrayBindingPos,
      0U,
      SCAST_U32(rough_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      rough_descs_image_infos.data(),
      nullptr,
      nullptr));
  
  eastl::vector<VkDescriptorImageInfo> norm_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::NORMAL,
      norm_descs_image_infos);

  // Normal textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kNormalTexturesArrayBindingPos,
      0U,
      SCAST_U32(norm_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      norm_descs_image_infos.data(),
      nullptr,
      nullptr));

  // Accumulation buffer
  VkDescriptorImageInfo accum_buff_img_info =
    accum_buffer_->image()->GetDescriptorImageInfo(nearest_sampler_);
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GENERIC],
      kAccumulationBufferBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
      &accum_buff_img_info,
      nullptr,
      nullptr));
  
  //// Accumulation ssao noise textures
  //VulkanTexture *ssao_noise_texture = texture_manager()->GetTextureByName("ssao_random_noise");
  //VKS_ASSERT(ssao_noise_texture != nullptr, "SSAO noise texture not found!");
  //eastl::array<VkDescriptorImageInfo, 3U> ssao_buffs_img_info = {
  //  ssao_buffer_->image()->GetDescriptorImageInfo(nearest_sampler_),
  //  ssao_noise_texture->image()->GetDescriptorImageInfo(nearest_sampler_repeat_),
  //  ssao_blur_buffer_->image()->GetDescriptorImageInfo(nearest_sampler_)
  //};
  //write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
  //    desc_sets_[SetTypes::GENERIC],
  //    kSSAOBuffersBindingPos,
  //    0U,
  //    SCAST_U32(ssao_buffs_img_info.size()),
  //    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
  //    ssao_buffs_img_info.data(),
  //    nullptr,
  //    nullptr));
  
  //// G Buffer
  //eastl::array<VkDescriptorImageInfo, GBtypes::num_items> g_buff_img_infos;
  //for (uint32_t i = 0U; i < GBtypes::num_items; i++) {
  //  g_buff_img_infos[i] =
  //    g_buffer_[i]->image()->GetDescriptorImageInfo(nearest_sampler_);
  //  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
  //      desc_sets_[SetTypes::GENERIC],
  //      kGBufferBaseBindingPos + i,
  //      0U,
  //      1U,
  //      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
  //      &g_buff_img_infos[i],
  //      nullptr,
  //      nullptr));
  //}

  // Update them 
  vkUpdateDescriptorSets(
      device.device(),
      SCAST_U32(write_desc_sets.size()),
      write_desc_sets.data(),
      0U,
      nullptr);
}

void FPlusRenderer::UpdatePVMatrices() {
  proj_mat_ = cam_->projection_mat();
  view_mat_ = cam_->view_mat();
  inv_proj_mat_ = glm::inverse(proj_mat_);
  inv_view_mat_ = glm::inverse(view_mat_);
}

void FPlusRenderer::CreateCommandBuffers(const VulkanDevice &device) {
  cmd_buffers_.resize(vulkan()->swapchain().GetNumImages());

  VkCommandBufferAllocateInfo cmd_buffer_allocate_info = {
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    nullptr,
    device.graphics_queue().cmd_pool,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    SCAST_U32(cmd_buffers_.size())
  };

  VK_CHECK_RESULT(vkAllocateCommandBuffers(
        device.device(),
        &cmd_buffer_allocate_info,
        cmd_buffers_.data()));

  cmd_buffer_allocate_info.commandBufferCount = 1U;
  VK_CHECK_RESULT(vkAllocateCommandBuffers(
    device.device(),
    &cmd_buffer_allocate_info,
    &cmd_buff_depth_prepass_));
  
  cmd_buffer_allocate_info.commandBufferCount = 1U;
  cmd_buffer_allocate_info.commandPool = device.compute_queue().cmd_pool;
  VK_CHECK_RESULT(vkAllocateCommandBuffers(
    device.device(),
    &cmd_buffer_allocate_info,
    &cmd_buff_compute_));
}

void FPlusRenderer::SetupGraphicsCommandBuffers(const VulkanDevice &device) {
  // Cache common settings to all command buffers 
  VkCommandBufferBeginInfo cmd_buff_begin_info =
    tools::inits::CommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
  cmd_buff_begin_info.pInheritanceInfo = nullptr;

  eastl::vector<VkClearValue> clear_values;
  eastl::vector<VkClearValue> clear_values_depth_prepass;
  VkClearValue clear_value;
  clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  clear_values.push_back(clear_value);
  clear_value.depthStencil = {1.f, 0U};
  clear_values.push_back(clear_value);
  clear_values_depth_prepass.push_back(clear_value);
  clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  clear_values.push_back(clear_value);
  //clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  //clear_values.push_back(clear_value);
  //clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  //clear_values.push_back(clear_value);

  // Record command buffers
  // First record the depth prepass command buffer
  VK_CHECK_RESULT(vkBeginCommandBuffer(
      cmd_buff_depth_prepass_, &cmd_buff_begin_info));

  depth_prepass_renderpass_->BeginRenderpass(
      cmd_buff_depth_prepass_,
      VK_SUBPASS_CONTENTS_INLINE,
      depth_prepass_framebuffer_.get(),
      {0U, 0U, cam_->viewport().width, cam_->viewport().height},
      SCAST_U32(clear_values_depth_prepass.size()),
      clear_values_depth_prepass.data());
  
  depth_prepass_material_->BindPipeline(cmd_buff_depth_prepass_, VK_PIPELINE_BIND_POINT_GRAPHICS);
  
  vkCmdBindDescriptorSets(
      cmd_buff_depth_prepass_,
      VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipe_layouts_[PipeLayoutTypes::GENERIC],
      0U,
      DescSetLayoutTypes::MODELS,
      desc_sets_.data(),
      0U,
      nullptr);
    
  for (eastl::vector<Model*>::iterator itor =
           registered_models_.begin();
         itor != registered_models_.end();
         ++itor) {
      (*itor)->BindVertexBuffer(cmd_buff_depth_prepass_);
      (*itor)->BindIndexBuffer(cmd_buff_depth_prepass_);
      (*itor)->RenderMeshesByMaterial(
          cmd_buff_depth_prepass_,
          pipe_layouts_[PipeLayoutTypes::GENERIC],
          DescSetLayoutTypes::MODELS);
    }
    
  depth_prepass_renderpass_->EndRenderpass(cmd_buff_depth_prepass_);
  VK_CHECK_RESULT(vkEndCommandBuffer(cmd_buff_depth_prepass_));

  uint32_t num_swapchain_images = vulkan()->swapchain().GetNumImages();
  for (uint32_t i = 0U; i < num_swapchain_images; i++) {
    VK_CHECK_RESULT(vkBeginCommandBuffer(
        cmd_buffers_[i], &cmd_buff_begin_info));

    shade_renderpass_->BeginRenderpass(
        cmd_buffers_[i],
        VK_SUBPASS_CONTENTS_INLINE,
        framebuffers_[i].get(),
        {0U, 0U, cam_->viewport().width, cam_->viewport().height},
        SCAST_U32(clear_values.size()),
        clear_values.data());

    shading_material_->BindPipeline(cmd_buffers_[i],
                                    VK_PIPELINE_BIND_POINT_GRAPHICS);

    vkCmdBindDescriptorSets(
        cmd_buffers_[i],
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipe_layouts_[PipeLayoutTypes::GENERIC],
        0U,
        DescSetLayoutTypes::MODELS,
        desc_sets_.data(),  
        0U,
        nullptr);

  for (eastl::vector<Model*>::iterator itor =
           registered_models_.begin();
         itor != registered_models_.end();
         ++itor) {
      (*itor)->BindVertexBuffer(cmd_buffers_[i]);
      (*itor)->BindIndexBuffer(cmd_buffers_[i]);
      (*itor)->RenderMeshesByMaterial(
          cmd_buffers_[i],
          pipe_layouts_[PipeLayoutTypes::GENERIC],
          DescSetLayoutTypes::MODELS);
    }

    //
    //// SSAO pass
    //renderpass_->NextSubpass(graphics_buffs[i], VK_SUBPASS_CONTENTS_INLINE);

    //g_ssao_material_->BindPipeline(graphics_buffs[i],
    //                                VK_PIPELINE_BIND_POINT_GRAPHICS);

    //fullscreenquad_->BindVertexBuffer(graphics_buffs[i]);  
    //fullscreenquad_->BindIndexBuffer(graphics_buffs[i]);  

    //vkCmdDrawIndexed(
    //    graphics_buffs[i],
    //    6U,
    //    1U,
    //    0U,
    //    0U,
    //    0U);
    //
    //// SSAO blur pass
    //renderpass_->NextSubpass(graphics_buffs[i], VK_SUBPASS_CONTENTS_INLINE);

    //g_ssao_blur_material_->BindPipeline(graphics_buffs[i],
    //                                VK_PIPELINE_BIND_POINT_GRAPHICS);

    //fullscreenquad_->BindVertexBuffer(graphics_buffs[i]);  
    //fullscreenquad_->BindIndexBuffer(graphics_buffs[i]);  

    //vkCmdDrawIndexed(
    //    graphics_buffs[i],
    //    6U,
    //    1U,
    //    0U,
    //    0U,
    //    0U);

    //// Light shading pass
    //renderpass_->NextSubpass(graphics_buffs[i], VK_SUBPASS_CONTENTS_INLINE);

    //g_shade_material_->BindPipeline(graphics_buffs[i],
    //                                VK_PIPELINE_BIND_POINT_GRAPHICS);

    //fullscreenquad_->BindVertexBuffer(graphics_buffs[i]);  
    //fullscreenquad_->BindIndexBuffer(graphics_buffs[i]);  

    //vkCmdDrawIndexed(
    //    graphics_buffs[i],
    //    6U,
    //    1U,
    //    0U,
    //    0U,
    //    0U);

    // Tonemapping pass
    shade_renderpass_->NextSubpass(cmd_buffers_[i], VK_SUBPASS_CONTENTS_INLINE);

    tonemap_material_->BindPipeline(cmd_buffers_[i],
                                      VK_PIPELINE_BIND_POINT_GRAPHICS);
    
    fullscreenquad_->BindVertexBuffer(cmd_buffers_[i]);  
    fullscreenquad_->BindIndexBuffer(cmd_buffers_[i]);  

    fullscreenquad_->BindVertexBuffer(cmd_buffers_[i]);
    fullscreenquad_->BindIndexBuffer(cmd_buffers_[i]);

    vkCmdDrawIndexed(
        cmd_buffers_[i],
        6U,
        1U,
        0U,
        0U,
        0U);

    shade_renderpass_->EndRenderpass(cmd_buffers_[i]);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmd_buffers_[i]));
  }
}

void FPlusRenderer::SetupComputeCommandBuffers(const VulkanDevice &device) {
  // Cache common settings to all command buffers 
  VkCommandBufferBeginInfo cmd_buff_begin_info =
    tools::inits::CommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
  cmd_buff_begin_info.pInheritanceInfo = nullptr;

  VK_CHECK_RESULT(vkBeginCommandBuffer(
      cmd_buff_compute_, &cmd_buff_begin_info));

  // Use a barrier to allow the buffers to be read by the compute pipeline
  eastl::array<VkBufferMemoryBarrier, 1U> barriers_before;
  barriers_before[0U] = tools::inits::BufferMemoryBarrier(
    VK_ACCESS_SHADER_READ_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    device.graphics_queue().index,
    device.compute_queue().index,
    light_idxs_buff_.buffer(),
    0U,
    light_idxs_buff_.size());

  vkCmdPipelineBarrier(
    cmd_buff_compute_,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0U,
    0, nullptr,
    barriers_before.size(),
    barriers_before.data(),
    0, nullptr);

  lights_cull_material_->BindPipeline(cmd_buff_compute_, VK_PIPELINE_BIND_POINT_COMPUTE);
  
  vkCmdBindDescriptorSets(
      cmd_buff_compute_,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipe_layouts_[PipeLayoutTypes::GENERIC],
      0U,
      DescSetLayoutTypes::MODELS,
      desc_sets_.data(),
      0U,
      nullptr);

  vkCmdDispatch(cmd_buff_compute_, kWidthInTiles, kHeightInTiles, 1U);

  // Use a barrier to allow the buffers to be read by the compute pipeline
  eastl::array<VkBufferMemoryBarrier, 1U> barriers_after;
  barriers_after[0U] = tools::inits::BufferMemoryBarrier(
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_ACCESS_SHADER_READ_BIT,
    device.compute_queue().index,
    device.graphics_queue().index,
    light_idxs_buff_.buffer(),
    0U,
    light_idxs_buff_.size());

  vkCmdPipelineBarrier(
    cmd_buff_compute_,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    0U,
    0, nullptr,
    barriers_after.size(),
    barriers_after.data(),
    0, nullptr);


  VK_CHECK_RESULT(vkEndCommandBuffer(cmd_buff_compute_));
}

void FPlusRenderer::SetupSamplers(const VulkanDevice &device) {
  // Create an aniso sampler
  VkSamplerCreateInfo sampler_create_info = tools::inits::SamplerCreateInfo(
      VK_FILTER_LINEAR,
      VK_FILTER_LINEAR,
      VK_SAMPLER_MIPMAP_MODE_LINEAR,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      0.f,
      VK_TRUE,
      device.physical_properties().limits.maxSamplerAnisotropy,
      VK_FALSE,
      VK_COMPARE_OP_NEVER,
      0.f,
      11.f,
      VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
      VK_FALSE);
  
  VK_CHECK_RESULT(vkCreateSampler(
      device.device(),
      &sampler_create_info,
      nullptr,
      &aniso_sampler_));

  // Create a nearest neighbour sampler
  sampler_create_info = tools::inits::SamplerCreateInfo(
      VK_FILTER_NEAREST,
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      0.f,
      VK_FALSE,
      0U,
      VK_FALSE,
      VK_COMPARE_OP_NEVER,
      0.f,
      1.f,
      VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
      VK_FALSE);

  VK_CHECK_RESULT(vkCreateSampler(
      device.device(),
      &sampler_create_info,
      nullptr,
      &nearest_sampler_));
  
  // Create a nearest neighbour sampler which repeats
  sampler_create_info = tools::inits::SamplerCreateInfo(
      VK_FILTER_NEAREST,
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      0.f,
      VK_FALSE,
      0U,
      VK_FALSE,
      VK_COMPARE_OP_NEVER,
      0.f,
      1.f,
      VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
      VK_FALSE);

  VK_CHECK_RESULT(vkCreateSampler(
      device.device(),
      &sampler_create_info,
      nullptr,
      &nearest_sampler_repeat_));
}
  
void FPlusRenderer::SetupMaterialPipelines(
    const VulkanDevice &device,
    const VertexSetup &store_vertex_setup) {
  eastl::vector<VertexElement> vtx_layout;
  vtx_layout.push_back(VertexElement(
        VertexElementType::POSITION,
        SCAST_U32(sizeof(glm::vec3)),
        VK_FORMAT_R32G32B32_SFLOAT));

  VertexSetup vertex_setup_quads(vtx_layout);

  // Setup depth prepass material
  eastl::unique_ptr<MaterialShader> depth_vert =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "passthrough.vert",
      "main",
      ShaderTypes::VERTEX);
  
  uint32_t num_materials = material_manager()->GetMaterialInstancesCount();
  
  eastl::unique_ptr<MaterialBuilder> builder_depth_prepass =
    eastl::make_unique<MaterialBuilder>(
    store_vertex_setup,
    "depth_prepass",
    pipe_layouts_[PipeLayoutTypes::GENERIC],
    depth_prepass_renderpass_->GetVkRenderpass(),
    VK_FRONT_FACE_COUNTER_CLOCKWISE,
    0U,
    cam_->viewport());
  builder_depth_prepass->AddShader(eastl::move(depth_vert));
  builder_depth_prepass->SetDepthTest(VK_COMPARE_OP_LESS);
  builder_depth_prepass->SetDepthWriteEnable(VK_TRUE);
  builder_depth_prepass->SetDepthTestEnable(VK_TRUE);

  depth_prepass_material_ =
    material_manager()->CreateMaterial(device, eastl::move(builder_depth_prepass)); 

  // Setup culling material
  eastl::unique_ptr<MaterialShader> culling_compute =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "light_culling.comp",
      "main",
      ShaderTypes::COMPUTE);
  
  uint32_t num_lights = lights_manager()->GetNumLights();
  culling_compute->AddSpecialisationEntry(
      kTileSizeSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &kTileSize);
  culling_compute->AddSpecialisationEntry(
      kNumLightsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_lights);
  //culling_compute->AddSpecialisationEntry(
  //    kMaxLightsPerTileSpecConstPos,
  //    SCAST_U32(sizeof(uint32_t)),
  //    &kMaxLightsPerTile);
  //culling_compute->AddSpecialisationEntry(
  //    kRasterWidthSpecConstPos,
  //    SCAST_U32(sizeof(uint32_t)),
  //    &kWindowWidth);
  //culling_compute->AddSpecialisationEntry(
  //    kRasterHeightSpecConstPos,
  //    SCAST_U32(sizeof(uint32_t)),
  //    &kWindowHeight);

  eastl::unique_ptr<MaterialBuilder> builder_culling =
    eastl::make_unique<MaterialBuilder>(
    "light_culling",
    pipe_layouts_[PipeLayoutTypes::GENERIC],
    cam_->viewport());
  
  builder_culling->AddShader(eastl::move(culling_compute));



  lights_cull_material_ =
    material_manager()->CreateMaterial(device, eastl::move(builder_culling)); 

  // Setup shading material
  eastl::unique_ptr<MaterialShader> shade_frag =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "fpshade.frag",
      "main",
      ShaderTypes::FRAGMENT);
  
  eastl::unique_ptr<MaterialShader> shade_vert =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "fpshade.vert",
      "main",
      ShaderTypes::VERTEX);
  
  shade_vert->AddSpecialisationEntry(
      kNumMaterialsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_materials);
  shade_vert->AddSpecialisationEntry(
      kNumLightsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_lights);
  shade_frag->AddSpecialisationEntry(
      kNumMaterialsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_materials);
  shade_frag->AddSpecialisationEntry(
      kNumLightsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_lights);

  eastl::unique_ptr<MaterialBuilder> builder_shade =
    eastl::make_unique<MaterialBuilder>(
    store_vertex_setup,
    "shade",
    pipe_layouts_[PipeLayoutTypes::GENERIC],
    shade_renderpass_->GetVkRenderpass(),
    VK_FRONT_FACE_COUNTER_CLOCKWISE,
    0U,
    cam_->viewport());
  
  builder_shade->AddColorBlendAttachment(
        VK_FALSE,
        VK_BLEND_FACTOR_ONE,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_OP_ADD,
        VK_BLEND_FACTOR_ONE,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_OP_ADD,
        0xf);
  
  float blend_constants[4U] = { 1.f, 1.f, 1.f, 1.f };
  builder_shade->AddColorBlendStateCreateInfo(
      VK_FALSE,
      VK_LOGIC_OP_SET,
      blend_constants);

  builder_shade->AddShader(eastl::move(shade_vert));
  builder_shade->AddShader(eastl::move(shade_frag));
  builder_shade->SetDepthTestEnable(VK_TRUE);
  builder_shade->SetDepthWriteEnable(VK_TRUE);
  builder_shade->SetDepthTest(VK_COMPARE_OP_LESS_OR_EQUAL);

  shading_material_ =
    material_manager()->CreateMaterial(device, eastl::move(builder_shade)); 

  // Setup tonemap material
  eastl::unique_ptr<MaterialShader> tone_frag =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "tonemapping.frag",
      "main",
      ShaderTypes::FRAGMENT);

  tone_frag->AddSpecialisationEntry(
    kTonemapExposureSpecConstPos,
    SCAST_U32(sizeof(float)),
    SCAST_CVOIDPTR(&kTonemapExposure));

  eastl::unique_ptr<MaterialShader> tone_vert =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "tonemapping.vert",
      "main",
      ShaderTypes::VERTEX);

  eastl::unique_ptr<MaterialBuilder> builder_tone =
    eastl::make_unique<MaterialBuilder>(
    vertex_setup_quads,
    "tonemapping",
    pipe_layouts_[PipeLayoutTypes::GENERIC],
    shade_renderpass_->GetVkRenderpass(),
    VK_FRONT_FACE_COUNTER_CLOCKWISE,
    1U,
    cam_->viewport());

  builder_tone->AddColorBlendAttachment(
      VK_FALSE,
      VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      VK_BLEND_OP_ADD,
      VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      VK_BLEND_OP_ADD,
      0xf);
  builder_tone->AddColorBlendStateCreateInfo(
      VK_FALSE,
      VK_LOGIC_OP_SET,
      blend_constants);
  builder_tone->AddShader(eastl::move(tone_vert));
  builder_tone->AddShader(eastl::move(tone_frag));

  tonemap_material_ =
    material_manager()->CreateMaterial(device, eastl::move(builder_tone)); 
  
  //// Setup SSAO material
  //eastl::unique_ptr<MaterialShader> ssao_gen_frag =
  //  eastl::make_unique<MaterialShader>(
  //    kBaseShaderAssetsPath + "ssao_gen.frag",
  //    "main",
  //    ShaderTypes::FRAGMENT);

  //eastl::unique_ptr<MaterialShader> ssao_gen_vert =
  //  eastl::make_unique<MaterialShader>(
  //    kBaseShaderAssetsPath + "g_shade.vert",
  //    "main",
  //    ShaderTypes::VERTEX);
  //
  //ssao_gen_frag->AddSpecialisationEntry(
  //    kSSAOKernelSizeSpecConstPos,
  //    SCAST_U32(sizeof(uint32_t)),
  //    &kSSAOKernelSize);
  //ssao_gen_frag->AddSpecialisationEntry(
  //    kSSAORadiusSizeSpecConstPos,
  //    SCAST_U32(sizeof(float)),
  //    &kSSAORadius);

  //eastl::unique_ptr<MaterialBuilder> builder_ssao =
  //  eastl::make_unique<MaterialBuilder>(
  //  vertex_setup_quads,
  //  "g_ssao",
  //  pipe_layouts_[PipeLayoutTypes::GPASS],
  //  renderpass_->GetVkRenderpass(),
  //  VK_FRONT_FACE_CLOCKWISE,
  //  1U,
  //  cam_->viewport());

  //builder_ssao->AddColorBlendAttachment(
  //    VK_FALSE,
  //    VK_BLEND_FACTOR_ONE,
  //    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
  //    VK_BLEND_OP_ADD,
  //    VK_BLEND_FACTOR_ONE,
  //    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
  //    VK_BLEND_OP_ADD,
  //    0xf);
  //builder_ssao->AddColorBlendStateCreateInfo(
  //    VK_FALSE,
  //    VK_LOGIC_OP_SET,
  //    blend_constants);
  //builder_ssao->AddShader(eastl::move(ssao_gen_vert));
  //builder_ssao->AddShader(eastl::move(ssao_gen_frag));

  //g_ssao_material_ =
  //  material_manager()->CreateMaterial(device, eastl::move(builder_ssao)); 
  //
  //// Setup SSAO blur material
  //eastl::unique_ptr<MaterialShader> ssao_blur_frag =
  //  eastl::make_unique<MaterialShader>(
  //    kBaseShaderAssetsPath + "blur.frag",
  //    "main",
  //    ShaderTypes::FRAGMENT);

  //eastl::unique_ptr<MaterialShader> ssao_blur_vert =
  //  eastl::make_unique<MaterialShader>(
  //    kBaseShaderAssetsPath + "g_shade.vert",
  //    "main",
  //    ShaderTypes::VERTEX);
  //
  //ssao_blur_frag->AddSpecialisationEntry(
  //    kSSAONoiseTextureSizeSpecConstPos,
  //    SCAST_U32(sizeof(uint32_t)),
  //    &kSSAONoiseTextureSize);

  //eastl::unique_ptr<MaterialBuilder> builder_ssao_blur =
  //  eastl::make_unique<MaterialBuilder>(
  //  vertex_setup_quads,
  //  "g_ssao_blur",
  //  pipe_layouts_[PipeLayoutTypes::GPASS],
  //  renderpass_->GetVkRenderpass(),
  //  VK_FRONT_FACE_CLOCKWISE,
  //  2U,
  //  cam_->viewport());

  //builder_ssao_blur->AddColorBlendAttachment(
  //    VK_FALSE,
  //    VK_BLEND_FACTOR_ONE,
  //    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
  //    VK_BLEND_OP_ADD,
  //    VK_BLEND_FACTOR_ONE,
  //    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
  //    VK_BLEND_OP_ADD,
  //    0xf);
  //builder_ssao_blur->AddColorBlendStateCreateInfo(
  //    VK_FALSE,
  //    VK_LOGIC_OP_SET,
  //    blend_constants);
  //builder_ssao_blur->AddShader(eastl::move(ssao_blur_vert));
  //builder_ssao_blur->AddShader(eastl::move(ssao_blur_frag));

  //g_ssao_blur_material_ =
  //  material_manager()->CreateMaterial(device, eastl::move(builder_ssao_blur));
}

void FPlusRenderer::SetupFullscreenQuad(const VulkanDevice &device){
  eastl::vector<VertexElement> vtx_layout;
  vtx_layout.push_back(VertexElement(
        VertexElementType::POSITION,
        SCAST_U32(sizeof(glm::vec3)),
        VK_FORMAT_R32G32B32_SFLOAT));

  VertexSetup vertex_setup_quads(vtx_layout);

  // Group all vertex data together
  ModelBuilder model_builder(
    vertex_setup_quads,
    desc_pool_);

  Vertex vtx;
  vtx.pos = { -1.f, 1.f, 0.f };
  model_builder.AddVertex(vtx);
  vtx.pos = { -1.f, -1.f, 0.f };
  model_builder.AddVertex(vtx);
  vtx.pos = { 1.f, -1.f, 0.f };
  model_builder.AddVertex(vtx);
  vtx.pos = { 1.f, 1.f, 0.f };
  model_builder.AddVertex(vtx);

  model_builder.AddIndex(0U);
  model_builder.AddIndex(2U);
  model_builder.AddIndex(1U);
  model_builder.AddIndex(0U);
  model_builder.AddIndex(3U);
  model_builder.AddIndex(2U);

  Mesh quad_mesh(
    0U,
    6U,
    0U,
    0U);

  model_builder.AddMesh(&quad_mesh);

  model_manager()->CreateModel(device, "fullscreenquad", model_builder,
                               &fullscreenquad_);
}

void FPlusRenderer::UpdateLights(eastl::vector<Light> &transformed_lights) {
  transformed_lights = lights_manager()->TransformLights(view_mat_);
}

void FPlusRenderer::ReloadAllShaders() {
  material_manager()->ReloadAllShaders(vulkan()->device());

  SetupGraphicsCommandBuffers(vulkan()->device());
}

} // namespace vks
