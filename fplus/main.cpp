#include <base_system.h>
#include <fplus_scene.h>
#include <EASTL/unique_ptr.h>
#include <EASTL/utility.h>

int main() {

  vks::Init();
  eastl::unique_ptr<vks::FPlusScene> scene =
    eastl::make_unique<vks::FPlusScene>();
  vks::Run(scene.get());
  vks::Shutdown();

  return 0;
}
