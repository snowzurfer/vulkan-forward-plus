#ifndef VKS_FPLUSSCENE
#define VKS_FPLUSSCENE

#include <scene.h>
#include <fplus_renderer.h>
#include <camera.h>
#include <camera_controller.h>

namespace vks {

const float kDefaultCameraSpeed = 80.f;
const float kDefaultCameraRotationSpeed = 50.f;

class FPlusScene : public Scene {
 public:
  FPlusScene();

 private:
  void DoInit();
  void DoRender(float delta_time);
  void DoUpdate(float delta_time);
  void DoShutdown(); 

  FPlusRenderer renderer_;
  szt::Camera cam_;
  szt::CameraController cam_controller_;

}; // class FPlusScene

} // namespace vks

#endif
