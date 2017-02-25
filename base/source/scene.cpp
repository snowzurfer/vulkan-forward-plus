#include <scene.h>

namespace vks {

Scene::Scene() {}

Scene::~Scene() {}

void Scene::Init() {
  return DoInit();
}

void Scene::Update(float delta_time) {
  DoUpdate(delta_time);
}

void Scene::Shutdown() {
  DoShutdown();
}

void Scene::Render(float delta_time) {
  DoRender(delta_time);
}

} // namespace vks
