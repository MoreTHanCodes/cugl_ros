#include <cugl_ros/apps/dynamic_fusion/dynfu_app.h>
#include <unistd.h>

namespace dynfu
{
static const char *glsl_shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

DynfuApp::DynfuApp(unsigned int refresh_delay,
                   unsigned int win_width,
                   unsigned int win_height)
    : cugl::VisualizerBase(refresh_delay),
      mesh_window0_(0),
      mesh_window1_(0),
      tex_window_(0),
      last_timestamp_(0),
      next_timestamp_(1000),
      fps_count_(0),
      mode_(PREVIEW),
      tex_mode_(TEX_LIVE),
      stop_(false),
      cam_view_(true),
      show_ref_mesh_(false),
      mesh_win_width_(win_width),
      mesh_win_height_(win_height),
      tex_win_width_(0),
      tex_win_height_(0),
      fps_(0),
      frame_count_(0),
      mouse_buttons_(0),
      select_start_(false),
      select_end_(false),
      mask_start_x_(0),
      mask_start_y_(0),
      mask_end_x_(0),
      mask_end_y_(0),
      valid_volume_(false),
      /*dynfu_thread_(),*/
      mesh_mutex_(),
      tex_mutex_(),
      surfel_mutex_(),
      cond_mesh_ready_(),
      cond_tex_ready_(),
      cond_surfel_ready_(),
      energy_init_(0.f),
      energy_final_(0.f),
      mesh_vertex_size_(0),
      new_mesh_vertex_size_(0),
      surfel_size_(0),
      gl_shader_(0),
      depth_trunc_(),
      depth_scaled_(),
      depth_filtered_(),
      vertex_image_(),
      normal_image_(),
      vertex_image_true_(),
      normal_image_true_(),
      dst_color_image_(),
      dst_texture_(0),
      dst_tex_bridge_(),
      mesh_vertex_array_(),
      mesh_normal_array_(),
      mesh_color_array_(),
      new_mesh_vertex_array_(),
      new_mesh_normal_array_(),
      new_mesh_color_array_(),
      surfel_vertex_array_(),
      surfel_normal_array_(),
      surfel_color_array_(),
      warped_mesh_vertex_array_(),
      warped_mesh_normal_array_(),
      warped_surfel_vertex_array_(),
      warped_surfel_normal_array_(),
      surfel_xyzrgb_data_(),
      mesh0_vertex_array_(),
      mesh0_normal_array_(),
      mesh0_color_array_(),
      mesh1_vertex_array_(),
      mesh1_normal_array_(),
      mesh1_color_array_(),
      active_surfel_flags_(),
      sum_buf_()
{
}

DynfuApp::~DynfuApp()
{
  // free();
}

bool DynfuApp::init(int argc, char **argv, ros::NodeHandle &nh)
{
  setInstance();

  // Use device with highest Gflops/s
  cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

  // Stream capture
  stream_capture_ptr_ = StreamCapture::Ptr(new StreamCapture(10));

  if (!stream_capture_ptr_->init())
  {
    std::cout << "Dynfu: Stream capture initialization failed!" << std::endl;
    return false;
  }

  stream_capture_ptr_->getIntrinsics(intrinsics_);

  if (argc != 2)
  {
    std::cout << "Dynfu: Invalid input format!" << std::endl;
    return false;
  }

  // Reading current configuration from the file
  {
    std::string curr_config_file(argv[1]);
    std::cout << "Dynfu: Reading current configuration from " << curr_config_file << std::endl;
    cv::FileStorage fs;
    fs.open(curr_config_file, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
      std::cout << "Dynfu: Can't open configuration file " << curr_config_file<< std::endl;
      return false;
    }

    fs["DynfuConfigure"] >> configure_;
    configure_.width = intrinsics_.width;
    configure_.height = intrinsics_.height;
    configure_.cx = intrinsics_.cx;
    configure_.cy = intrinsics_.cy;
    configure_.fx = intrinsics_.fx;
    configure_.fy = intrinsics_.fy;
    configure_.depth_scale = intrinsics_.depth_scale;

    std::cout << configure_ << std::endl;
    fs.release();
    std::cout << "Dynfu: Reading done." << std::endl;
  }

  // Saving current configuration to the file
  {
    std::string prev_config_file = "./prev_configure.xml";
    std::cout << "Dynfu: Saving current configuration in " << prev_config_file << std::endl;
    cv::FileStorage fs(prev_config_file, cv::FileStorage::WRITE);
    fs << "DynfuConfigure" << configure_;
    fs.release();
    std::cout << "Dynfu: Saving done." << std::endl;
  }

  // Initialize OpenGL context, so we can properly set the GL for CUDA
  if (!initGL(&argc, argv))
  {
    std::cout << "Dynfu: OpenGL initialization failed!" << std::endl;
    return false;
  }

  gpu::Roi roi;
  roi.start = make_int2(30, 20);
  roi.end = make_int2(intrinsics_.width - 30, intrinsics_.height - 20);

  gpu::IcpParamList icp_param;
  icp_param.dist_thres = configure_.dist_thres;
  icp_param.angle_thres = configure_.angle_thres;
  icp_param.view_angle_thres = configure_.view_angle_thres;

  // Deformable Voxel Hierarchy
  ref_volume_ptr_ = DeformVoxelHierarchy::Ptr(
                 new DeformVoxelHierarchy(
                     icp_param,
                     intrinsics_,
                     roi,
                     configure_.trunc_dist,
                     configure_.volume_length,
                     configure_.voxel_block_inner_dim_shift,
                     configure_.voxel_block_dim_shift,
                     configure_.node_block_dim_shift_base,
                     configure_.w_fit,
                     configure_.w_fit_huber,
                     configure_.w_reg,
                     configure_.w_reg_huber));

  cur_volume_ptr_ = DeformVoxelHierarchy::Ptr(
                 new DeformVoxelHierarchy(
                     icp_param,
                     intrinsics_,
                     roi,
                     configure_.trunc_dist,
                     configure_.volume_length,
                     configure_.voxel_block_inner_dim_shift,
                     configure_.voxel_block_dim_shift,
                     configure_.node_block_dim_shift_base,
                     configure_.w_fit,
                     configure_.w_fit_huber,
                     configure_.w_reg,
                     configure_.w_reg_huber));

  // Set initial affine transformation
  Eigen::Matrix<float, 3, 3> rotation;
  rotation = Eigen::AngleAxisf(configure_.rz, Eigen::Vector3f::UnitZ()) *
             Eigen::AngleAxisf(configure_.ry, Eigen::Vector3f::UnitY()) *
             Eigen::AngleAxisf(configure_.rx, Eigen::Vector3f::UnitX());
  affine_init_.linear() = rotation;
  affine_init_.translation() = Eigen::Vector3f(configure_.tx, configure_.ty, configure_.tz);

  affine_view_ = affine_init_;
  rotate_ = make_float3(0.f, 0.f, 0.f);
  translate_ = make_float3(0.f, 0.f, 0.f);

  allocMemory(intrinsics_.width, intrinsics_.height);

  reset();

  // ros::init(argc, argv, "dynfu_node");

  ros_pub_ = nh.advertise<sensor_msgs::PointCloud2>("dynfu_cloud", 1);

  return true;
}

void DynfuApp::startMainLoop()
{
  stream_capture_ptr_->start();

  sleep(2); // wait until RGB-D camera work stably

  // dynfu_thread_ = std::thread(&DynfuApp::startPipeline, this);

  glutMainLoop();
}

bool DynfuApp::getSurfelDataHost(size_t request_size, float *host_data)
{
  std::unique_lock<std::mutex> lock(surfel_mutex_);
  MilliSeconds milli_seconds(50);
  const auto ready = [this]() { return true; };

  if (cond_surfel_ready_.wait_for(lock, milli_seconds, ready))
  {
    if (request_size == surfel_size_)
    {
      surfel_xyzrgb_data_.download(host_data, 0, request_size * 8);
    }

    return true;
  }

  return false;
}

void DynfuApp::display()
{
  if (!stop_)
  {
    int cur_win = glutGetWindow();

    if (cur_win)
    {
      if (cur_win == mesh_window0_)
      {
        runPipelineOnce();

        renderMeshWin0();
      }

      if (cur_win == mesh_window1_)
        renderMeshWin1();

      if (cur_win == tex_window_)
        renderTexWin();
    }
  }

  // double time_stamp;
  // ColorFrameHandle color_frame_handle;
  // DepthFrameHandle depth_frame_handle;

  // if (stream_capture_ptr_->fetchFrame(&time_stamp, &color_frame_handle, &depth_frame_handle, 500))
  // {
  //   preprocess(*depth_frame_handle,
  //              *color_frame_handle);

  //   gpu::bindImageTextures(intrinsics_.width,
  //                          intrinsics_.height,
  //                          vertex_image_,
  //                          normal_image_,
  //                          depth_scaled_,
  //                          *color_frame_handle);

  //   bool success = true;

  //   switch (mode_)
  //   {
  //     case (PREVIEW):
  //       reset();
  //       success = initDeformVoxelHierarchy();
  //       break;

  //     case (TRACKING_AND_FUSION):
  //       trackRigidMotion();
  //       frame_count_++;
  //       trackNonRigidMotion();
  //       fuseVolume();
  //       break;

  //     case (TRACKING_ONLY):
  //       trackRigidMotion();
  //       frame_count_++;
  //       trackNonRigidMotion();
  //       break;
  //   }

  //   if (success)
  //   {
  //     success = fetchSurface();
  //   }

  //   if (success)
  //   {
  //     warpSurface();
  //   }

  //   gpu::unbindImageTextures();

  //   if (success && mode_ != TRACKING_ONLY)
  //     success = updateDeformVoxelHierarchy();

  //   if (success)
  //   {
  //     renderAllWins();
  //   }

  //   std::cout << '\r' << "DynfuApp: Processed frame [" << frame_count_ << "]" << std::flush;

  //   if (!success && mode_ != PREVIEW)
  //     reset();

  //   glutSetWindow(main_window_);

  //   int fps = updateFps(time_stamp);

  //   std::stringstream ss;
  //   ss << "Shape Tracking and Reconstruction" << "  fps(" << fps << ")  voxel_block_size(" << ref_volume_ptr_->getVoxelBlockSize() << ")  node(" << ref_volume_ptr_->getNodeSize() << ")  mesh(" << mesh_vertex_size_ << ")";
  //   glutSetWindowTitle(ss.str().c_str());
  // }
}

void DynfuApp::keyboard(unsigned char key, int x, int y)
{
  switch (key)
  {
    case (27):
      exit();
      break;

    case (32):
      if (mode_ == TRACKING_ONLY)
      {
        mode_ = PREVIEW;
      }
      else if (mode_ == TRACKING_AND_FUSION)
      {
        mode_ = TRACKING_ONLY;
      }
      else if (mode_ == PREVIEW)
      {
        mode_ = TRACKING_AND_FUSION;
      }
      break;

    case ('t'):
      if (tex_mode_ == TEX_FUSION)
      {
        tex_mode_ = TEX_NONE;
      }
      else if (tex_mode_ == TEX_NONE)
      {
        tex_mode_ = TEX_LIVE;
      }
      else if (tex_mode_ == TEX_LIVE)
      {
        tex_mode_ = TEX_FUSION;
      }
      break;

    case ('v'):
      cam_view_ = !cam_view_;
      if (cam_view_)
      {
        rotate_ = make_float3(0.f, 0.f, 0.f);
        translate_ = make_float3(0.f, 0.f, 0.f);
      }
      break;

    case ('r'):
      show_ref_mesh_ = !show_ref_mesh_;
      break;

    case ('s'):
      stream_capture_ptr_->saveStream();
      break;
  }
}

void DynfuApp::mouse(int button, int state, int x, int y)
{
  int cur_win = glutGetWindow();

  if (cur_win)
  {
    if (cur_win == mesh_window0_ || cur_win == mesh_window1_)
    {
      if (state == GLUT_DOWN)
      {
        mouse_buttons_ |= 1<<button;
      }
      else if (state == GLUT_UP)
      {
        mouse_buttons_ = 0;
      }

      mouse_old_x_ = x;
      mouse_old_y_ = y;
    }
    else if (cur_win == tex_window_)
    {
      if (button == GLUT_LEFT_BUTTON)
      {
        if (state == GLUT_DOWN)
        {
          if (!select_start_ && !select_end_)
          {
            select_start_ = true;
            select_end_ = false;
            mask_start_x_ = x;
            mask_start_y_ = y;
          }
        }
        else if (state == GLUT_UP)
        {
          if (select_start_ && !select_end_)
          {
            select_end_ = true;
            mask_end_x_ = x;
            mask_end_y_ = y;
          }
        }
      }
      else if (button == GLUT_RIGHT_BUTTON)
      {
        select_start_ = false;
        select_end_ = false;
      }
    }
  }
}

void DynfuApp::motion(int x, int y)
{
  float dx = static_cast<float>(x - mouse_old_x_);
  float dy = static_cast<float>(y - mouse_old_y_);

  if (!cam_view_)
  {
    if (mouse_buttons_ == 1)
    {
      rotate_.x += dy * 0.2f;
      rotate_.y += dx * 0.2f;
    }
    else if (mouse_buttons_ == 2)
    {
      translate_.x += dx * 0.01f;
      translate_.y += dy * 0.01f;
    }
    else if (mouse_buttons_ == 4)
    {
      translate_.z += dy * 0.01f;
    }
  }

  mouse_old_x_ = x;
  mouse_old_y_ = y;
}

void DynfuApp::reshape(int w, int h)
{
  int cur_win = glutGetWindow();

  if (cur_win)
  {
    if (cur_win == mesh_window0_ || cur_win == mesh_window1_)
    {
      mesh_win_width_ = w;
      mesh_win_height_ = h;

      glutSetWindow(mesh_window0_);
      glutReshapeWindow(mesh_win_width_, mesh_win_height_);
      setProjection(mesh_win_width_, mesh_win_height_);

      glutSetWindow(mesh_window1_);
      glutReshapeWindow(mesh_win_width_, mesh_win_height_);
      setProjection(mesh_win_width_, mesh_win_height_);
    }
    else if (cur_win == tex_window_)
    {
      tex_win_width_ = w;
      tex_win_height_ = h;

      glutSetWindow(tex_window_);
      glutReshapeWindow(tex_win_width_, tex_win_height_);
      setProjection(tex_win_width_, tex_win_height_);
    }
  }
}

bool DynfuApp::initGL(int *argc, char **argv)
{
  // Create GL context
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

  // Create window for mesh visualization

  // mesh window 0
  glutInitWindowSize(mesh_win_width_, mesh_win_height_);
  mesh_window0_ = glutCreateWindow("Multiple-frame Reconstruction");

  if (!isGLVersionSupported(2, 0))
  {
    std::cout << "Dynfu: Support for necessary OpenGL extensions missing." << std::endl;
    return false;
  }

  // init window background
  glClearColor(0.f, 0.f, 0.f, 1.f);

  if (!compileASMShader(GL_FRAGMENT_PROGRAM_ARB, glsl_shader_code))
  {
    std::cout << "Dynfu: ASM shader compilation failed. " << std::endl;
    return false;
  } 
  glutReportErrors();

  // Register callbacks
  glutDisplayFunc(displayWrapper);
  glutKeyboardFunc(keyboardWrapper);
  glutReshapeFunc(reshapeWrapper);
  glutTimerFunc(refresh_delay_, timerEventWrapper, 0);
  glutMouseFunc(mouseWrapper);
  glutMotionFunc(motionWrapper);

  glutReportErrors();

  // draw mesh window 1
  glutInitWindowSize(mesh_win_width_, mesh_win_height_);
  mesh_window1_ = glutCreateWindow("Single-frame Reconstruction");

  if (configure_.draw_mesh_win1)
  {
    // init window background
    glClearColor(0.f, 0.f, 0.f, 1.f);

    // Register callbacks
    glutDisplayFunc(displayWrapper);
    glutReshapeFunc(reshapeWrapper);
    glutTimerFunc(refresh_delay_, timerEventWrapper, 0);
  }

  glutReportErrors();

  // draw tex window 0
  glutInitWindowSize(tex_win_width_, tex_win_height_);
  tex_window_ = glutCreateWindow("Raw RGB frame");
  tex_win_width_ = intrinsics_.width;
  tex_win_height_ = intrinsics_.height;

  if (configure_.draw_tex_win0)
  {
    // init window background
    glClearColor(0.f, 0.f, 0.f, 1.f);

    // Register callbacks
    glutDisplayFunc(displayWrapper);
    glutReshapeFunc(reshapeWrapper);
    glutTimerFunc(refresh_delay_, timerEventWrapper, 0);
    glutMouseFunc(mouseWrapper);

  }

  glutReportErrors();

  return true;
}

void DynfuApp::allocMemory(int image_width, int image_height)
{
  affines_.reserve(36000);
  res_sqrts_.reserve(36000);
  vo_flags_.reserve(36000);

  depth_trunc_.alloc((size_t)image_width, (size_t)image_height);
  depth_scaled_.alloc((size_t)image_width, (size_t)image_height);
  depth_filtered_.alloc((size_t)image_width, (size_t)image_height);
  vertex_image_.alloc((size_t)image_width, (size_t)image_height);
  normal_image_.alloc((size_t)image_width, (size_t)image_height);

  vertex_image_true_.alloc((size_t)image_width, (size_t)image_height);
  normal_image_true_.alloc((size_t)image_width, (size_t)image_height);

  mesh_vertex_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE));
  mesh_normal_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE));
  mesh_color_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE));
  warped_mesh_vertex_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE));
  warped_mesh_normal_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE));

  new_mesh_vertex_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE));
  new_mesh_normal_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE));
  new_mesh_color_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE));

  surfel_vertex_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);
  surfel_normal_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);
  surfel_color_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);
  warped_surfel_vertex_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);
  warped_surfel_normal_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);

  surfel_xyzrgb_data_.alloc((size_t)(gpu::MAX_TRIANGLES_SIZE * 8));

  active_surfel_flags_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);

  // use host buffer
  sum_buf_.alloc(27, true);

  // use OpenGL VBO
  if (mesh_window0_)
  {
    glutSetWindow(mesh_window0_);

    mesh0_vertex_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);
    mesh0_normal_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);
    mesh0_color_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);
  }

  // use OpenGL VBO
  if (mesh_window1_)
  {
    glutSetWindow(mesh_window1_);

    mesh1_vertex_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);
    mesh1_normal_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);
    mesh1_color_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);
  }

  // use OpenGL Texture
  if (tex_window_)
  {
    glutSetWindow(tex_window_); // important

    dst_color_image_.alloc((size_t)intrinsics_.width, (size_t)intrinsics_.height);

    if(!dst_texture_) glGenTextures(1, &dst_texture_);
    glBindTexture(GL_TEXTURE_2D, dst_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, intrinsics_.width, intrinsics_.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    dst_tex_bridge_.create((size_t)intrinsics_.width, (size_t)intrinsics_.height, dst_texture_, cugl::TextureBridge::WRITE_ONLY);
  }
}

void DynfuApp::free()
{
  depth_trunc_.free();
  depth_scaled_.free();
  depth_filtered_.free();
  vertex_image_.free();
  normal_image_.free();

  vertex_image_true_.free();
  normal_image_true_.free();

  mesh_vertex_array_.free();
  mesh_normal_array_.free();
  mesh_color_array_.free();
  surfel_vertex_array_.free();
  surfel_normal_array_.free();
  surfel_color_array_.free();

  new_mesh_vertex_array_.free();
  new_mesh_normal_array_.free();
  new_mesh_color_array_.free();

  warped_mesh_vertex_array_.free();
  warped_mesh_normal_array_.free();
  warped_surfel_vertex_array_.free();
  warped_surfel_normal_array_.free();

  surfel_xyzrgb_data_.free();

  active_surfel_flags_.free();

  sum_buf_.free();

  dst_color_image_.free();
  dst_tex_bridge_.free();

  if(dst_texture_)
  {
    glDeleteTextures(1, &dst_texture_);
    dst_texture_ = 0;
  }

  mesh0_vertex_array_.free();
  mesh0_normal_array_.free();
  mesh0_color_array_.free();

  mesh1_vertex_array_.free();
  mesh1_normal_array_.free();
  mesh1_color_array_.free();
}

void DynfuApp::reset()
{
  if (frame_count_)
  {
    std::cout << std::endl << RED << "Dynfu: Reset!" << RESET << std::endl;
  }

  mode_ = PREVIEW;

  frame_count_ = 0;

  affines_.clear();

  affines_.push_back(affine_init_);

  res_sqrts_.clear();

  res_sqrts_.push_back(0.f);

  vo_flags_.clear();

  vo_flags_.push_back(0);

  ref_volume_ptr_->reset();
}

void DynfuApp::exit()
{
  stop_ = true;

  stream_capture_ptr_->stop();

  saveResiduals();

  free();

  if (mesh_window0_)
    glutDestroyWindow(mesh_window0_);

  if (mesh_window1_)
    glutDestroyWindow(mesh_window1_);

  if (tex_window_)
    glutDestroyWindow(tex_window_);

  // if(dynfu_thread_.joinable())
  // {
  //   dynfu_thread_.join();
  // }

  // free();
}

void DynfuApp::updateFps(double timestamp)
{
  if(timestamp != last_timestamp_)
  {
    last_timestamp_ = timestamp;

    ++fps_count_;
    if(timestamp >= next_timestamp_)
    {
      fps_ = fps_count_;
      fps_count_ = 0;
      next_timestamp_ += 1000;
    }
  }
}

bool DynfuApp::compileASMShader(GLenum program_type, const char *code)
{
  glGenProgramsARB(1, &gl_shader_);
  glBindProgramARB(program_type, gl_shader_);
  glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

  GLint error_pos;
  glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

  if (error_pos != -1)
  {
    const GLubyte *error_string;
    error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);

    std::cout << "Dynfu:: Program error at postion: " << (int)error_pos << " " << error_string << std::endl;

    return false;
  }

  return true;
}

void DynfuApp::runPipelineOnce()
{
  double time_stamp;
  ColorFrameHandle color_frame_handle;
  DepthFrameHandle depth_frame_handle;

  if (stream_capture_ptr_->fetchFrame(&time_stamp, &color_frame_handle, &depth_frame_handle, 500))
  {
    preprocess(*depth_frame_handle,
               *color_frame_handle);

    gpu::bindImageTextures(intrinsics_.width,
                           intrinsics_.height,
                           vertex_image_,
                           normal_image_,
                           depth_scaled_,
                           *color_frame_handle);

    bool success = true;

    switch (mode_)
    {
      case (PREVIEW):
        reset();
        success &= initDeformVoxelHierarchy(ref_volume_ptr_);
        break;

      case (TRACKING_AND_FUSION):
        trackRigidMotion();
        frame_count_++;
        trackNonRigidMotion();
        fuseVolume();
        break;

      case (TRACKING_ONLY):
        trackRigidMotion();
        frame_count_++;
        trackNonRigidMotion();
        break;
    }

    cur_volume_ptr_->reset();
    success &= initDeformVoxelHierarchy(cur_volume_ptr_);

    std::unique_lock<std::mutex> lock(mesh_mutex_);

    if (success)
    {
      success = fetchSurface(ref_volume_ptr_);

      new_mesh_vertex_size_ = cur_volume_ptr_->fetchIsoSurface(new_mesh_vertex_array_,
                                                               new_mesh_normal_array_,
                                                               new_mesh_color_array_);

      success &= (new_mesh_vertex_size_ > 0);
    }

    lock.unlock();

    cond_mesh_ready_.notify_one();

    if (success)
    {
      warpSurface();
    }

    gpu::unbindImageTextures();

    if (success && mode_ != TRACKING_ONLY)
      success = updateDeformVoxelHierarchy();

    std::cout << '\r' << "DynfuApp: Processed frame [" << frame_count_ << "]" << std::flush;

    if (!success && mode_ != PREVIEW)
      reset();

    updateFps(time_stamp);
  }
}

void DynfuApp::preprocess(const cugl::DepthImage16u &depth_raw,
                          const cugl::ColorImage8u &color_raw)
{
  float2 depth_thres = make_float2(configure_.depth_min,
                                   configure_.depth_max);

  float4 color_thres = make_float4(configure_.h_min,
                                   configure_.h_max,
                                   configure_.s_min,
                                   configure_.s_max);

  bool draw_mask = select_start_ && select_end_;
  int2 mask_start = make_int2(mask_start_x_, mask_start_y_);
  int2 mask_end = make_int2(mask_end_x_, mask_end_y_);

  // [calculate groud truth of vertex map and normal map]
  gpu::truncateDepth(depth_thres,
                     color_thres,
                     false,
                     mask_start,
                     mask_end,
                     intrinsics_,
                     depth_raw,
                     color_raw,
                     depth_trunc_,
                     depth_scaled_);

  gpu::bilateralFilter(intrinsics_.width,
                       intrinsics_.height,
                       depth_trunc_,
                       depth_filtered_);

  gpu::createVertexImage(intrinsics_,
                         depth_filtered_,
                         vertex_image_true_);

  gpu::createNormalImage(intrinsics_.width,
                         intrinsics_.height,
                         vertex_image_true_,
                         normal_image_true_);
  // [end]

  gpu::truncateDepth(depth_thres,
                     color_thres,
                     draw_mask,
                     mask_start,
                     mask_end,
                     intrinsics_,
                     depth_raw,
                     color_raw,
                     depth_trunc_,
                     depth_scaled_);

  gpu::bilateralFilter(intrinsics_.width,
                       intrinsics_.height,
                       depth_trunc_,
                       depth_filtered_);

  gpu::createVertexImage(intrinsics_,
                         depth_filtered_,
                         vertex_image_);

  gpu::createNormalImage(intrinsics_.width,
                         intrinsics_.height,
                         vertex_image_,
                         normal_image_);

  std::unique_lock<std::mutex> lock(tex_mutex_);

  if (false/*tex_mode_ == TEX_NONE*/)
  {
    depth_thres = make_float2(0.f,
                              2.f);

    gpu::depthToColor(depth_thres,
                      intrinsics_.depth_scale,
                      intrinsics_.width,
                      intrinsics_.height,
                      depth_raw,
                      dst_color_image_);
  }
  else
  {
    gpu::convertColor(intrinsics_.width,
                      intrinsics_.height,
                      draw_mask,
                      mask_start,
                      mask_end,
                      color_raw,
                      dst_color_image_);
  }

  lock.unlock();

  cond_tex_ready_.notify_one();
}

void DynfuApp::trackRigidMotion()
{
  // Set the initial guess of current rigid transformation
  Eigen::Affine3f T_curr = affines_[frame_count_];
  Eigen::Matrix<float, 3, 3> R_curr = T_curr.linear();
  Eigen::Vector3f t_curr = T_curr.translation();

  size_t surfel_vertex_size = mesh_vertex_size_ / 3;

  // Solve rigid motion tracking based on an ICP algorithm
  for (int iter = 0; iter < 10; ++iter)
  {
    cugl::Affine3d T_curr_device(T_curr.data());

    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 1> b;

    // Formulate ICP as Gauss-Newton Optimization
    ref_volume_ptr_->formIcpGaussNewtonOpt(T_curr_device,
                                    surfel_vertex_size,
                                    warped_surfel_vertex_array_,
                                    warped_surfel_normal_array_,
                                    sum_buf_,
                                    A.data(),
                                    b.data());

    double det = A.determinant();

    if (fabs(det) < 1e-15 || std::isnan(det))
    {
      std::cout << std::endl << RED << "Dynfu: Invalid det(A) in rigid motion tracking!" << RESET << std::endl;
      break;
    }

    // Solve Gauss-Newton Optimization
    Eigen::Matrix<float, 6, 1> result = A.llt().solve(b).cast<float>();

    float rx = result(0);
    float ry = result(1);
    float rz = result(2);
    Eigen::Matrix<float, 3, 3> R_inc;
    R_inc = Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()) *
            Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()) *
            Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX());

    Eigen::Vector3f t_inc = result.tail<3>();

    // compose
    R_curr = R_inc * R_curr;
    t_curr = R_inc * t_curr + t_inc;
    T_curr.linear() = R_curr;
    T_curr.translation() = t_curr;
  }

  affines_.push_back(T_curr);
}

void DynfuApp::trackNonRigidMotion()
{
  Eigen::Affine3f T_curr = affines_[frame_count_];

  cugl::Affine3d T_curr_device(T_curr.data());

  size_t surfel_vertex_size = mesh_vertex_size_ / 3;

  gpu::bindMeshTextures(surfel_vertex_array_,
                        surfel_normal_array_);

  bool success = ref_volume_ptr_->initDeformEnergyTerms(T_curr_device,
                                      surfel_vertex_size,
                                      energy_init_);

  if (!success)
  {
    std::cout << std::endl << RED << "DynfuApp: deform energy terms initialization failed!" << RESET << std::endl;
  }
  else
  {
    float energy_last = energy_init_;
    float energy_curr = energy_init_;

    // Solve non-rigid motion tracking based on the Gauss-Newton method
    for (int iter = 0; iter < configure_.icp_iters; iter++)
    {
      ref_volume_ptr_->formDetGaussNewtonOpt();

      float damping_old = 0.f;
      float damping_new = configure_.lm_damping;

      for (int lm_iter = 0; lm_iter < configure_.lm_iters; lm_iter++)
      {
        bool valid_result = ref_volume_ptr_->solveDetGaussNewtonOpt(damping_new - damping_old); // solve with damping factor

        if (!valid_result)
        {
          std::cout << std::endl << RED << "Dynfu: Invalid result occurs in PCG solver!" << RESET << std::endl;
          damping_old = damping_new;
          damping_new *= configure_.lm_beta;
        }
        else
        {
          // perform backtracking line search
          const float beta = 0.8f;
          const float step_size_end = 0.01f * beta;
          float step_size = 1.f;

          ref_volume_ptr_->initTransformSearch();

          while (step_size > step_size_end)
          {
            ref_volume_ptr_->runTransformSearchOnce(step_size);

            ref_volume_ptr_->updateDeformEnergyTerms(T_curr_device,
                                              surfel_vertex_size,
                                              energy_curr);

            if (energy_curr < energy_last)
            {
              energy_last = energy_curr;
              break;
            }
            step_size *= beta;
          } // step_size > step_size_end
        } //valid pcg solver result

        if (energy_last == energy_curr)
        {
          damping_old = damping_new;
          damping_new *= configure_.lm_alpha;
          break;
        }
        else
        {
          damping_old = damping_new;
          damping_new *= configure_.lm_beta;
        }
      } // lm
    } // icp

    energy_final_ = energy_last;
  }

  gpu::unbindMeshTextures();
}

bool DynfuApp::initDeformVoxelHierarchy(DeformVoxelHierarchy::Ptr volume_ptr)
{
  Eigen::Affine3f T_curr = affines_[0];

  Eigen::Matrix<float, 3, 3> R_curr = T_curr.linear();
  Eigen::Vector3f t_curr = T_curr.translation();

  Eigen::Matrix<float, 3, 3> R_curr_inv = R_curr.inverse();
  Eigen::Vector3f t_curr_inv = -(R_curr_inv * t_curr);

  Eigen::Affine3f T_curr_inv;
  T_curr_inv.linear() = R_curr_inv;
  T_curr_inv.translation() = t_curr_inv;

  cugl::Affine3d T_curr_device(T_curr.data());
  cugl::Affine3d T_curr_inv_device(T_curr_inv.data());

  float4 color_thres = make_float4(configure_.h_min,
                                   configure_.h_max,
                                   configure_.s_min,
                                   configure_.s_max);

  bool success = volume_ptr->init(T_curr_device,
                                  T_curr_inv_device,
                                  color_thres);

  if (!success)
    std::cout << std::endl << RED << "DynfuApp: Deformable Voxel Hierarchy initilization failed!" << RESET << std::endl;

  return success;
}

void DynfuApp::fuseVolume()
{
  Eigen::Affine3f T_curr = affines_[frame_count_];
  cugl::Affine3d T_curr_device(T_curr.data());

  float4 color_thres = make_float4(configure_.h_min,
                                   configure_.h_max,
                                   configure_.s_min,
                                   configure_.s_max);

  ref_volume_ptr_->fuseVolume(T_curr_device, false, color_thres);
}

bool DynfuApp::fetchSurface(DeformVoxelHierarchy::Ptr volume_ptr)
{
  mesh_vertex_size_ = volume_ptr->fetchIsoSurface(mesh_vertex_array_,
                                                  mesh_normal_array_,
                                                  mesh_color_array_,
                                                  surfel_vertex_array_,
                                                  surfel_normal_array_,
                                                  surfel_color_array_);

  return (mesh_vertex_size_ != 0);
}

void DynfuApp::warpSurface()
{
  gpu::bindImageTrueTextures(intrinsics_.width,
                             intrinsics_.height,
                             vertex_image_true_,
                             normal_image_true_);

  Eigen::Affine3f T_curr = affines_[frame_count_];
  cugl::Affine3d T_curr_device(T_curr.data());

  float res_sqrt = 0.f;

  ref_volume_ptr_->warpSurface(T_curr_device,
                               mesh_vertex_size_,
                               mesh_vertex_array_,
                               mesh_normal_array_,
                               mesh_color_array_,
                               warped_mesh_vertex_array_,
                               warped_mesh_normal_array_,
                               warped_surfel_vertex_array_,
                               warped_surfel_normal_array_,
                               active_surfel_flags_,
                               res_sqrt);

  res_sqrts_.push_back(res_sqrt);

  vo_flags_.push_back(int(select_start_ && select_end_));

  gpu::unbindImageTrueTextures();

  surfel_size_ = mesh_vertex_size_ / 3;

  // std::unique_lock<std::mutex> lock(surfel_mutex_);

  if (surfel_size_ > 0)
  {
    gpu::combineVertexAndColorData(surfel_size_,
                                   T_curr_device,
                                   warped_surfel_vertex_array_,
                                   surfel_color_array_,
                                   surfel_xyzrgb_data_);

    pcl::PointCloud<pcl::PointXYZRGB> cloud;

    cloud.width = surfel_size_;
    cloud.height = 1;
    cloud.points.resize(cloud.width * cloud.height);

    surfel_xyzrgb_data_.download(reinterpret_cast<float *>(cloud.points.data()), 0, surfel_size_ * 8);

    sensor_msgs::PointCloud2 output;

    pcl::toROSMsg(cloud, output);
    output.header.frame_id = "map";
    output.header.stamp = ros::Time::now();

    ros_pub_.publish(output);

    ros::spinOnce();
  }

  // lock.unlock();

  // cond_surfel_ready_.notify_one();
}

bool DynfuApp::updateDeformVoxelHierarchy()
{
  bool success = ref_volume_ptr_->update(mesh_vertex_size_ / 3,
                                         surfel_vertex_array_,
                                         active_surfel_flags_);

  return success;
}

// void DynfuApp::renderAllWins()
// {
//   renderMainWin();
// 
//   renderSubWinMesh0();
// 
//   renderSubWinMesh1();
// 
//   renderSubWinTex0();
// }
// 
// void DynfuApp::renderMainWin()
// {
//   glutSetWindow(main_window_);
// 
//   glClearColor(0.1, 0.2f, 0.3f, 1.f);
// 
//   glClear(GL_COLOR_BUFFER_BIT);
// 
//   glutSwapBuffers();
// }

void DynfuApp::renderMeshWin0()
{
  glutSetWindow(mesh_window0_);

  if (mesh_vertex_size_ > 0)
  {
    mesh0_vertex_array_.map();
    mesh0_normal_array_.map();
    mesh0_color_array_.map();

    if (show_ref_mesh_)
    {
      cudaMemcpy((void *)mesh0_vertex_array_.getDevicePtr(),
                 (void *)mesh_vertex_array_.getDevicePtr(),
                 mesh_vertex_size_ * sizeof(float4),
                 cudaMemcpyDeviceToDevice);

      cudaMemcpy((void *)mesh0_normal_array_.getDevicePtr(),
                 (void *)mesh_normal_array_.getDevicePtr(),
                 mesh_vertex_size_ * sizeof(float4),
                 cudaMemcpyDeviceToDevice);
    }
    else
    {
      cudaMemcpy((void *)mesh0_vertex_array_.getDevicePtr(),
                 (void *)warped_mesh_vertex_array_.getDevicePtr(),
                 mesh_vertex_size_ * sizeof(float4),
                 cudaMemcpyDeviceToDevice);

      cudaMemcpy((void *)mesh0_normal_array_.getDevicePtr(),
                 (void *)warped_mesh_normal_array_.getDevicePtr(),
                 mesh_vertex_size_ * sizeof(float4),
                 cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy((void *)mesh0_color_array_.getDevicePtr(),
               (void *)mesh_color_array_.getDevicePtr(),
               mesh_vertex_size_ * sizeof(float4),
               cudaMemcpyDeviceToDevice);

    mesh0_vertex_array_.unmap();
    mesh0_normal_array_.unmap();
    mesh0_color_array_.unmap();

    // clear information from the last draw
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // move light with camera
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // light 0
    float ambient[]   = { 0.1f, 0.1f, 0.1f, 1.0f };
    float diffuse[]   = { 0.6f, 0.6f, 0.6f, 1.0f };
    float specular[]  = { 0.2f, 0.2f, 0.2f, 1.0f };
    float light_pos[] = { 0.3f, 0.5f, -0.8f, 0.0f };
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

    glEnable(GL_LIGHT0);

    // material
    if (tex_mode_ == TEX_NONE)
    {
      glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    }
    else
    {
      glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, ambient);
      glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 25.f);

      glEnable(GL_COLOR_MATERIAL);
    }

    glEnable(GL_DEPTH_TEST);
    // glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHTING);

    // modelview matrix
    glRotatef(180.0f, 1.f, 0.f, 0.f);

    if (cam_view_) // use camera view
      affine_view_ = affines_[frame_count_];

    glMultMatrixf(affine_view_.data());

    const float offset = configure_.volume_length / 2.0f;
    glTranslatef(offset, offset, offset);

    glTranslatef(translate_.x, translate_.y, translate_.z);
    glRotatef(rotate_.x, 1.0, 0.0, 0.0);
    glRotatef(rotate_.y, 0.0, -1.0, 0.0);

    // draw mesh
    this->drawMesh(mesh_window0_);

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    // draw the camera frustum
    if (!cam_view_)
    {
      // draw the TSDF volume
      glColor3f(0.9f, 0.9f, 0.9f);
      glutWireCube(configure_.volume_length);

      // const float offset = configure_.volume_length / 2.0f;
      // Eigen::Affine3f T_curr = affines_[frame_count_];
      // Eigen::Matrix<float, 3, 3> R_curr = T_curr.linear();
      // Eigen::Vector3f t_curr = T_curr.translation();

      // Eigen::Vector3f cam_pos = Eigen::Vector3f(-offset, -offset, -offset) - R_curr.inverse() * t_curr;
      // glTranslatef(cam_pos(0), cam_pos(1), cam_pos(2));
      // glutSolidSphere(0.04, 20, 10);
    }

    glutSwapBuffers();
    glutReportErrors();
  }
  else
  {
    // clear information from the last draw
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glutSwapBuffers();
    glutReportErrors();
  }

  std::stringstream ss;
  ss << "fps(" << fps_ << ")  voxel_block_size(" << ref_volume_ptr_->getVoxelBlockSize() << ")  node(" << ref_volume_ptr_->getNodeSize() << ")  mesh(" << mesh_vertex_size_ << ")";
  glutSetWindowTitle(ss.str().c_str());
}

void DynfuApp::renderMeshWin1()
{
  std::unique_lock<std::mutex> lock(mesh_mutex_);
  MilliSeconds milli_seconds(10);
  const auto ready = [this]() { return true; };

  if (cond_mesh_ready_.wait_for(lock, milli_seconds, ready))
  {
    glutSetWindow(mesh_window1_);

    if (new_mesh_vertex_size_ > 0)
    {
      mesh1_vertex_array_.map();
      mesh1_normal_array_.map();
      mesh1_color_array_.map();

      cudaMemcpy((void *)mesh1_vertex_array_.getDevicePtr(),
                 (void *)new_mesh_vertex_array_.getDevicePtr(),
                 new_mesh_vertex_size_ * sizeof(float4),
                 cudaMemcpyDeviceToDevice);

      cudaMemcpy((void *)mesh1_normal_array_.getDevicePtr(),
                 (void *)new_mesh_normal_array_.getDevicePtr(),
                 new_mesh_vertex_size_ * sizeof(float4),
                 cudaMemcpyDeviceToDevice);

      cudaMemcpy((void *)mesh1_color_array_.getDevicePtr(),
                 (void *)new_mesh_color_array_.getDevicePtr(),
                 new_mesh_vertex_size_ * sizeof(float4),
                 cudaMemcpyDeviceToDevice);

      mesh1_vertex_array_.unmap();
      mesh1_normal_array_.unmap();
      mesh1_color_array_.unmap();

      // clear information from the last draw
      glClearColor(0.f, 0.f, 0.f, 1.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      // move light with camera
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      // light 0
      float ambient[]   = { 0.1f, 0.1f, 0.1f, 1.0f };
      float diffuse[]   = { 0.6f, 0.6f, 0.6f, 1.0f };
      float specular[]  = { 0.2f, 0.2f, 0.2f, 1.0f };
      float light_pos[] = { 0.3f, 0.5f, -0.8f, 0.0f };
      glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
      glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
      glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
      glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

      glEnable(GL_LIGHT0);

      // material
      if (tex_mode_ == TEX_NONE)
      {
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
      }
      else
      {
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, ambient);
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 25.f);

        glEnable(GL_COLOR_MATERIAL);
      }

      glEnable(GL_DEPTH_TEST);
      // glEnable(GL_NORMALIZE);
      glEnable(GL_LIGHTING);

      // modelview matrix
      glRotatef(180.0f, 1.f, 0.f, 0.f);

      // if (cam_view_) // use camera view
      //   affine_view_ = affines_[frame_count_];

      // glMultMatrixf(affine_view_.data());
      glMultMatrixf(affine_init_.data());

      const float offset = configure_.volume_length / 2.0f;
      glTranslatef(offset, offset, offset);

      glTranslatef(translate_.x, translate_.y, translate_.z);
      glRotatef(rotate_.x, 1.0, 0.0, 0.0);
      glRotatef(rotate_.y, 0.0, -1.0, 0.0);

      // draw mesh
      this->drawMesh(mesh_window1_);

      glDisable(GL_LIGHTING);
      glDisable(GL_DEPTH_TEST);

      glutSwapBuffers();
      glutReportErrors();
    }
    else
    {
      // clear information from the last draw
      glClearColor(0.f, 0.f, 0.f, 1.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glutSwapBuffers();
      glutReportErrors();
    }
  }
}

void DynfuApp::renderTexWin()
{
  std::unique_lock<std::mutex> lock(tex_mutex_);
  MilliSeconds milli_seconds(10);
  const auto ready = [this]() { return true; };

  if (cond_tex_ready_.wait_for(lock, milli_seconds, ready))
  {
    glutSetWindow(tex_window_); // important

    dst_tex_bridge_.map();

    cudaArray *tex_ptr = dst_tex_bridge_.getCudaArrayPtr();

    cudaMemcpy2DToArray(tex_ptr, 0, 0,
                        (void *)dst_color_image_.getDevicePtr(),
                        dst_color_image_.getPitch(),
                        dst_color_image_.getWidth() * sizeof(uchar4),
                        dst_color_image_.getHeight(),
                        cudaMemcpyDeviceToDevice);

    dst_tex_bridge_.unmap();

    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    this->drawTexture(dst_texture_);
  }
}

void DynfuApp::drawMesh(int win_num)
{
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);

  // mesh representation
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  // set vertex & normal buffer
  if (win_num == mesh_window0_)
  {
    glBindBuffer(GL_ARRAY_BUFFER, mesh0_vertex_array_.getVbo());
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, mesh0_normal_array_.getVbo());
    glNormalPointer(GL_FLOAT, sizeof(float)*4, 0);

    // set color buffer
    if (tex_mode_ != TEX_NONE)
    {
      glEnableClientState(GL_COLOR_ARRAY);
      glBindBuffer(GL_ARRAY_BUFFER, mesh0_color_array_.getVbo());
      glColorPointer(4, GL_FLOAT, 0, 0);
    }

    // draw triangles
    glDrawArrays(GL_TRIANGLES, 0, mesh_vertex_size_);
  }
  else if (win_num == mesh_window1_)
  {
    glBindBuffer(GL_ARRAY_BUFFER, mesh1_vertex_array_.getVbo());
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, mesh1_normal_array_.getVbo());
    glNormalPointer(GL_FLOAT, sizeof(float)*4, 0);

    // set color buffer
    if (tex_mode_ != TEX_NONE)
    {
      glEnableClientState(GL_COLOR_ARRAY);
      glBindBuffer(GL_ARRAY_BUFFER, mesh1_color_array_.getVbo());
      glColorPointer(4, GL_FLOAT, 0, 0);
    }

    // draw triangles
    glDrawArrays(GL_TRIANGLES, 0, new_mesh_vertex_size_);
  }

  // // draw triangles
  // glDrawArrays(GL_TRIANGLES, 0, mesh_vertex_size_);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  // reset buffer array
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void DynfuApp::drawTexture(GLuint texture)
{
  glBindTexture(GL_TEXTURE_2D, texture);
  glEnable(GL_TEXTURE_2D);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, (GLfloat)intrinsics_.width, 0.0, (GLfloat)intrinsics_.height, -1.0, 1.0);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glBegin(GL_QUADS);
  glTexCoord2f(0.0, 0.0);
  glVertex2f(0.0, intrinsics_.height);

  glTexCoord2f(1.0, 0.0);
  glVertex2f(intrinsics_.width, intrinsics_.height);

  glTexCoord2f(1.0, 1.0);
  glVertex2f(intrinsics_.width, 0.0);

  glTexCoord2f(0.0, 1.0);
  glVertex2f(0.0, 0.0);
  glEnd();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glDisable(GL_TEXTURE_2D);

  glutSwapBuffers();
  glutReportErrors();
}

void DynfuApp::setProjection(int w, int h)
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, static_cast<float>(w) / static_cast<float>(h), 0.1, 10.0);

  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);
}

void DynfuApp::saveResiduals()
{
  std::string file_name = "res_array.txt";

  std::ofstream os(file_name.c_str());

  if (os.is_open())
  {
    std::cout << "DynfuApp: Saving results in " << file_name << " ... " << std::endl;

    for (int i = 0; i < frame_count_; i++)
    {
      os << vo_flags_[i] << " " << res_sqrts_[i] << std::endl;
    }

    // pcl::PointCloud<pcl::PointXYZRGB> cloud;

    // cloud.width = 2;
    // cloud.height = 1;
    // cloud.points.resize(cloud.width * cloud.height);

    // cloud.points[0].x = 0.1f;
    // cloud.points[0].y = 0.2f;
    // cloud.points[0].z = 0.3f;
    // cloud.points[0].rgb = 0.5f;

    // cloud.points[1].x = 1.1f;
    // cloud.points[1].y = 1.2f;
    // cloud.points[1].z = 1.3f;
    // cloud.points[1].rgb = 1.5f;

    // float *data_ptr = reinterpret_cast<float *>(cloud.points.data());

    // for (int i = 0; i < 2; i++)
    // {
    //   os << "point[" << i << "] :" << std::endl;

    //   for (int j = 0; j < 8; j++)
    //   {
    //     os << data_ptr[i * 8 + j] << " ";
    //   }

    //   os << std::endl;
    // }

    std::cout << "DynfuApp: Done." << std::endl;
  }
  else
  {
    std::cout << "DynfuApp: Can't open file to save results!" << std::endl;
  }
}

} // namespace

int main(int argc, char **argv)
{
  ros::init(argc, argv, "dynfu_node");

  ros::NodeHandle nh;

  dynfu::DynfuApp app(17, 520, 520);

  if(app.init(argc, argv, nh))
  {
    app.startMainLoop();
  }

  return 0;
}
