#ifndef __CUGL_APPS_DYNFU_DEPTH_VIEWER_H__
#define __CUGL_APPS_DYNFU_DEPTH_VIEWER_H__

#define RESET "\033[0m"
#define RED   "\033[31m"
#define GREEN "\033[32m"

#include "image_capture.h"
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

namespace dynfu
{
class DynfuDepthViewer : public cugl::VisualizerBase
{
 public:
  DynfuDepthViewer(unsigned int refresh_delay = 33,
            unsigned int win_width = 640,
            unsigned int win_height = 480);

  virtual ~DynfuDepthViewer();

  virtual bool init(int argc, char **argv);

  virtual void startMainLoop();

 protected:
  // Rendering callbacks
  virtual void display();
  virtual void keyboard(unsigned char key, int x, int y);
  virtual void mouse(int button, int state, int x, int y);
  virtual void motion(int x, int y);
  virtual void reshape(int w, int h);

 private:
  typedef ImageCapture::ColorFrameHandle ColorFrameHandle;
  typedef ImageCapture::DepthFrameHandle DepthFrameHandle;

  unsigned int win_width_, win_height_;
  int frame_count_;
  bool first_frame_;
  std::atomic<bool> pause_;

  int mouse_old_x_, mouse_old_y_;
  int mouse_buttons_;
  float3 rotate_;
  float3 translate_;

  float energy_init_;
  float energy_final_;

  ImageCapture::Ptr image_capture_ptr_;
  DeformVoxelHierarchy::Ptr dvh_ptr_;

  gpu::Intrinsics intrinsics_;
  DynfuConfigure configure_;

  Eigen::Affine3f affine_init_;
  std::vector<Eigen::Affine3f> affines_;

  cugl::DepthImage16u depth_trunc_;
  cugl::DepthImage32f depth_scaled_;
  cugl::DepthImage16u depth_filtered_;
  cugl::VertexImage vertex_image_;
  cugl::NormalImage normal_image_;

  cugl::ColorImage8u4c depth_2_color_image_;
  GLuint dst_texture_;
  cugl::TextureBridge dst_tex_bridge_;

  GLuint gl_shader_;
  size_t mesh_vertex_size_;

  cugl::VertexArray mesh_vertex_array_;
  cugl::NormalArray mesh_normal_array_;
  cugl::VertexArray surfel_vertex_array_;
  cugl::NormalArray surfel_normal_array_;

  cugl::VertexArray warped_mesh_vertex_array_;
  cugl::NormalArray warped_mesh_normal_array_;
  cugl::VertexArray warped_surfel_vertex_array_;
  cugl::NormalArray warped_surfel_normal_array_;

  cugl::DeviceArray<uint> active_surfel_flags_;

  cugl::DeviceArray<float> sum_buf_;

  bool initGL(int *argc, char **argv);

  void allocMemory(int image_width, int image_height);

  void free();

  void reset();

  void exit();

  bool compileASMShader(GLenum program_type, const char *code);

  // *****************************
  // Core algorithm components
  // *****************************
  void preprocess(const cugl::DepthImage16u &depth_raw);

  void trackRigidMotion();

  void trackNonRigidMotion();

  bool initDeformVoxelHierarchy();

  void fuseVolume();

  bool fetchSurface();

  void warpSurface();

  bool updateDeformVoxelHierarchy();

  void renderScene();

  void displayTexture(GLuint texture);

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace dynfu

#endif /* __CUGL_APPS_DYNFU_DEPTH_VIEWER_H__ */
