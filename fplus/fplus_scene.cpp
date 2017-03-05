#include <fplus_scene.h>
#include <base_system.h>
#include <model_manager.h>
#include <vulkan_base.h>
#include <model.h>
#include <logger.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/postprocess.h>
#include <vertex_setup.h>
#include <EASTL/vector.h>
#include <frustum.h>

namespace vks {

extern const int32_t kWindowWidth = 1280U;
extern const int32_t kWindowHeight = 720U;
extern const char *kWindowName = "vksagres-fplus";

FPlusScene::FPlusScene()
    : Scene(),
      renderer_(),
      cam_() {}

void FPlusScene::DoInit() {
  input_manager()->SetCursorMode(window(), szt::MouseCursorMode::DISABLED);

  szt::Viewport viewport;
  viewport.x = 0U;
  viewport.y = 0U;
  viewport.width = kWindowWidth;
  viewport.height = kWindowHeight;
  szt::Frustum frustum(
      0.2f,
      1000.f,
      40.f,
      static_cast<float>(viewport.width) / static_cast<float>(viewport.height));
  cam_.Init(viewport, frustum);

  cam_controller_.Init(window(), input_manager(), kDefaultCameraSpeed,
                       kDefaultCameraRotationSpeed);

  lights_manager()->CreateLight(
    glm::vec3(100.f, 100.f, 100.f),
    glm::vec3(100.f, 100.f, 100.f),
    glm::vec3(-60.f, 10.f, 20.f),
    100.f);

  lights_manager()->CreateLight(
    glm::vec3(100.f, 100.f, 100.f),
    glm::vec3(10.f, 10.f, 10.f),
    glm::vec3(40.f, 12.f, 17.f),
    1000.f);

  // Setup the vertex layout of the model to be passed
  eastl::vector<VertexElement> vtx_layout;
  vtx_layout.push_back(VertexElement(
        VertexElementType::POSITION,
        SCAST_U32(sizeof(glm::vec3)),
        VK_FORMAT_R32G32B32_SFLOAT));
  vtx_layout.push_back(VertexElement(
        VertexElementType::NORMAL,
        SCAST_U32(sizeof(glm::vec3)),
        VK_FORMAT_R32G32B32_SFLOAT));
  vtx_layout.push_back(VertexElement(
        VertexElementType::UV,
        SCAST_U32(sizeof(glm::vec2)),
        VK_FORMAT_R32G32_SFLOAT));
  vtx_layout.push_back(VertexElement(
        VertexElementType::BITANGENT,
        SCAST_U32(sizeof(glm::vec3)),
        VK_FORMAT_R32G32B32_SFLOAT));
  vtx_layout.push_back(VertexElement(
        VertexElementType::TANGENT,
        SCAST_U32(sizeof(glm::vec3)),
        VK_FORMAT_R32G32B32_SFLOAT));

  VertexSetup vertex_setup(vtx_layout);

  renderer_.Init(&cam_);

  Model*nanosuit = nullptr;
  model_manager()->LoadOtherModel(
      vulkan()->device(),
      STR(ASSETS_FOLDER) "models/nanosuit/nanosuit.obj",
      STR(ASSETS_FOLDER) "models/nanosuit/",
        aiProcess_CalcTangentSpace |
          aiProcess_GenSmoothNormals |
          aiProcess_Triangulate |
          aiProcess_JoinIdenticalVertices |
          aiProcess_ConvertToLeftHanded,
      vertex_setup,
      &nanosuit);

  renderer_.RegisterModel(*nanosuit, vertex_setup);

  //Model *crate = nullptr;
  //model_manager()->LoadOtherModel(
  //    vulkan()->device(),
  //    kBaseModelAssetsPath + "Crate/Crate1.obj",
  //    kBaseModelAssetsPath + "Crate/",
  //    aiProcess_Triangulate |
  //      aiProcess_GenNormals |
  //      aiProcess_CalcTangentSpace,
  //    vertex_setup,
  //    RendererType::DEFERRED,
  //    &crate);

  //crate->SetModelMatrixForAllMeshes(
  //    glm::translate(glm::mat4(1.f), glm::vec3(0.f, -1.f, 0.f)) *
  //    glm::scale(glm::mat4(1.f), glm::vec3(100.f, 1.f, 100.f)));

  //renderer_.RegisterModel(*crate);

}

void FPlusScene::DoRender(float delta_time) {
  renderer_.PreRender();
  renderer_.Render();
  renderer_.PostRender();
}

void FPlusScene::DoUpdate(float delta_time) {
  cam_controller_.Update(&cam_, delta_time);

  // Reload shaders
  if (input_manager()->IsKeyPressed(GLFW_KEY_R)) {
    renderer_.ReloadAllShaders();
  }
}

void FPlusScene::DoShutdown() {
  renderer_.Shutdown();
}

} // namespace vks
