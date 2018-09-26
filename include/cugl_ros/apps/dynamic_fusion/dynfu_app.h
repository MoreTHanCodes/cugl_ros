#ifndef __CUGL_APPS_DYNFU_APP_H__
#define __CUGL_APPS_DYNFU_APP_H__

#define RESET "\033[0m"
#define RED   "\033[31m"
#define GREEN "\033[32m"

#include "stream_capture.h"
#include "dynfu_configure.h"
#include "deform_voxel_hierarchy.h"

#include <cugl_ros/visualization/visualizer_base.h>

// dirty workaround of confliction between Eigen and X11
#ifdef Success
  #undef Success
#endif
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>

// ros & pcl
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

namespace dynfu
{
class DynfuApp : public cugl::VisualizerBase
{
 public:
  DynfuApp(unsigned int refresh_delay = 33,
           unsigned int win_width = 640,
           unsigned int win_height = 480);

  virtual ~DynfuApp();

  virtual bool init(int argc, char **argv, ros::NodeHandle &nh);

  virtual void startMainLoop();

  size_t getSurfelSize() const { return surfel_size_; }

  bool getSurfelDataHost(size_t request_size, float *data_host);

  bool is_stopped() const { return (stop_ == true); }

 protected:
  // Rendering callbacks
  virtual void display();
  virtual void keyboard(unsigned char key, int x, int y);
  virtual void mouse(int button, int state, int x, int y);
  virtual void motion(int x, int y);
  virtual void reshape(int w, int h);

  // static void texWinMouseWrapper(int button, int state, int x, int y);
  // void texWinMouse(int button, int state, int x, int y);

 private:
  enum Mode
  {
    PREVIEW,             // Tracking(OFF), Fusion(OFF)
    TRACKING_AND_FUSION, // Tracking(ON),  Fusion(ON)
    TRACKING_ONLY        // Tracking(ON),  Fusion(OFF)
  };

  enum TexMode
  {
    TEX_NONE,
    TEX_LIVE,   // fetch texture from live image
    TEX_FUSION  // fetch texture from fused volume
  };

  typedef StreamCapture::ColorFrameHandle ColorFrameHandle;
  typedef StreamCapture::DepthFrameHandle DepthFrameHandle;
  typedef StreamCapture::MilliSeconds MilliSeconds;

  int mesh_window0_, mesh_window1_, tex_window_;
  double last_timestamp_, next_timestamp_;
  int fps_count_;
  float3 rotate_;
  float3 translate_;
  Eigen::Affine3f affine_view_;

  std::atomic<Mode> mode_;
  std::atomic<TexMode> tex_mode_;
  std::atomic<bool> stop_;
  std::atomic<bool> cam_view_;
  std::atomic<bool> show_ref_mesh_;
  std::atomic<unsigned int> mesh_win_width_;
  std::atomic<unsigned int> mesh_win_height_;
  std::atomic<unsigned int> tex_win_width_;
  std::atomic<unsigned int> tex_win_height_;
  std::atomic<int> fps_;
  std::atomic<int> frame_count_;
  std::atomic<int> mouse_old_x_, mouse_old_y_;
  std::atomic<int> mouse_buttons_;
  std::atomic<bool> select_start_, select_end_;
  std::atomic<int> mask_start_x_, mask_start_y_, mask_end_x_, mask_end_y_;
  // std::atomic<float3> rotate_;
  // std::atomic<float3> translate_;
  // std::atomic<Eigen::Affine3f> affine_view_;
  std::atomic<size_t> mesh_vertex_size_;
  std::atomic<size_t> new_mesh_vertex_size_;
  std::atomic<size_t> surfel_size_;
  std::atomic<bool> valid_volume_;

  std::thread dynfu_thread_;
  std::mutex mesh_mutex_, tex_mutex_, surfel_mutex_;
  std::condition_variable cond_mesh_ready_, cond_tex_ready_, cond_surfel_ready_;

  float energy_init_;
  float energy_final_;

  StreamCapture::Ptr stream_capture_ptr_;
  DeformVoxelHierarchy::Ptr ref_volume_ptr_;
  DeformVoxelHierarchy::Ptr cur_volume_ptr_;

  gpu::Intrinsics intrinsics_;
  DynfuConfigure configure_;

  Eigen::Affine3f affine_init_;
  std::vector<Eigen::Affine3f> affines_;
  std::vector<float> res_sqrts_;
  std::vector<int> vo_flags_;

  cugl::DepthImage16u depth_trunc_;
  cugl::DepthImage32f depth_scaled_;
  cugl::DepthImage16u depth_filtered_;
  cugl::VertexImage vertex_image_;
  cugl::NormalImage normal_image_;

  cugl::VertexImage vertex_image_true_;
  cugl::NormalImage normal_image_true_;

  cugl::ColorImage8u4c dst_color_image_;
  GLuint dst_texture_;
  cugl::TextureBridge dst_tex_bridge_;

  GLuint gl_shader_;
  // size_t mesh_vertex_size_;

  cugl::VertexArray mesh_vertex_array_;
  cugl::NormalArray mesh_normal_array_;
  cugl::ColorArray  mesh_color_array_;
  cugl::VertexArray surfel_vertex_array_;
  cugl::NormalArray surfel_normal_array_;
  cugl::DeviceArray<float> surfel_color_array_;

  cugl::VertexArray warped_mesh_vertex_array_;
  cugl::NormalArray warped_mesh_normal_array_;
  cugl::VertexArray warped_surfel_vertex_array_;
  cugl::NormalArray warped_surfel_normal_array_;

  cugl::DeviceArray<float> surfel_xyzrgb_data_;

  cugl::VertexArray new_mesh_vertex_array_;
  cugl::NormalArray new_mesh_normal_array_;
  cugl::ColorArray  new_mesh_color_array_;

  cugl::VertexArray mesh0_vertex_array_;
  cugl::NormalArray mesh0_normal_array_;
  cugl::ColorArray  mesh0_color_array_;

  cugl::VertexArray mesh1_vertex_array_;
  cugl::NormalArray mesh1_normal_array_;
  cugl::ColorArray  mesh1_color_array_;

  cugl::DeviceArray<uint> active_surfel_flags_;

  cugl::DeviceArray<float> sum_buf_;

  // ros::NodeHandle ros_nh_;
  ros::Publisher ros_pub_;

  bool initGL(int *argc, char **argv);

  void allocMemory(int image_width, int image_height);

  void free();

  void reset();

  void exit();

  void updateFps(double timestamp);

  bool compileASMShader(GLenum program_type, const char *code);

  // *****************************
  // Core algorithm components
  // *****************************
  void runPipelineOnce();

  void preprocess(const cugl::DepthImage16u &depth_raw,
                  const cugl::ColorImage8u &color_raw);

  void trackRigidMotion();

  void trackNonRigidMotion();

  bool initDeformVoxelHierarchy(DeformVoxelHierarchy::Ptr volume_ptr);

  void fuseVolume();

  bool fetchSurface(DeformVoxelHierarchy::Ptr volume_ptr);

  void warpSurface();

  bool updateDeformVoxelHierarchy();

  // void renderAllWins();

  // void renderMainWin();

  void renderMeshWin0();

  void renderMeshWin1();

  void renderTexWin();

  void drawMesh(int win_num);

  void drawTexture(GLuint texture);

  void setProjection(int w, int h);

  void saveResiduals();

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace

#endif /* __CUGL_APPS_DYNFU_APP_H__ */
