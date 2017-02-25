#ifndef VKS_SCENE
#define VKS_SCENE

namespace vks {

class Scene {
 public:
  Scene();
  virtual ~Scene();

  void Init();
  void Update(float delta_time);
  void Render(float delta_time);
  void Shutdown();

 private:
  virtual void DoUpdate(float delta_time) = 0;
  virtual void DoRender(float delta_time) = 0;
  virtual void DoInit() = 0;
  virtual void DoShutdown() = 0;
  
}; // class Scene

} // namespace vks

#endif
