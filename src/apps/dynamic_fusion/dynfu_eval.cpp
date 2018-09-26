#include <cugl_ros/apps/dynamic_fusion/dynfu_eval.h>
#include <unistd.h>

namespace dynfu
{
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

DynfuEval::DynfuEval(unsigned int refresh_delay,
                   unsigned int win_width,
                   unsigned int win_height)
    : cugl::VisualizerBase(refresh_delay),
      win_width_(win_width),
      win_height_(win_height),
      frame_count_(0),
      first_frame_(true),
      pause_(false),
      cam_view_(true),
      mouse_buttons_(0),
      energy_init_(0.f),
      energy_final_(0.f),
      mesh_vertex_size_(0),
      gl_shader_(0),
      depth_trunc_(),
      depth_scaled_(),
      depth_filtered_(),
      vertex_image_(),
      normal_image_(),
      mesh_vertex_array_(),
      mesh_normal_array_(),
      surfel_vertex_array_(),
      surfel_normal_array_(),
      warped_mesh_vertex_array_(),
      warped_mesh_normal_array_(),
      warped_surfel_vertex_array_(),
      warped_surfel_normal_array_(),
      active_surfel_flags_(),
      sum_buf_()
{
}

DynfuEval::~DynfuEval()
{
  // free();
}

bool DynfuEval::init(int argc, char **argv)
{
  setInstance();

  // Initialize OpenGL context, so we can properly set the GL for CUDA
  if (!initGL(&argc, argv))
  {
    std::cout << "Dynfu: OpenGL initialization failed!" << std::endl;
    return false;
  }

  // Use device with highest Gflops/s
  cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

  if (argc != 2)
  {
    std::cout << "Dynfu: Invalid input format!" << std::endl;
    return false;
  }

  std::string folder(argv[1]);

  if (folder[folder.size()-1] != '/')
    folder.push_back('/');

  // Reading current configuration from the file
  {
    std::string curr_config_file = folder + "dynfu_configure.xml";
    std::cout << "Dynfu: Reading current configuration from " << curr_config_file << std::endl;
    cv::FileStorage fs;
    fs.open(curr_config_file, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
      std::cout << "Dynfu: Can't open configuration file " << curr_config_file<< std::endl;
      return false;
    }

    fs["DynfuConfigure"] >> configure_;

    std::cout << configure_ << std::endl;
    fs.release();
    std::cout << "Dynfu: Reading done." << std::endl;
  }

  // Image capture
  image_capture_ptr_ = ImageCapture::Ptr(new ImageCapture(folder, 0, configure_.width, configure_.height));

  if (!image_capture_ptr_->init())
  {
    std::cout << "Dynfu: Image capture initialization failed!" << std::endl;
    return false;
  }

  intrinsics_.width = configure_.width;
  intrinsics_.height = configure_.height;
  intrinsics_.cx = configure_.cx;
  intrinsics_.cy = configure_.cy;
  intrinsics_.fx = configure_.fx;
  intrinsics_.fy = configure_.fy;
  intrinsics_.depth_scale = configure_.depth_scale;

  gpu::Roi roi;
  roi.start = make_int2(3, 2);
  roi.end = make_int2(intrinsics_.width - 3, intrinsics_.height - 2);

  gpu::IcpParamList icp_param;
  icp_param.dist_thres = configure_.dist_thres;
  icp_param.angle_thres = configure_.angle_thres;
  icp_param.view_angle_thres = configure_.view_angle_thres;

  // Deformable Voxel Hierarchy
  dvh_ptr_ = DeformVoxelHierarchy::Ptr(
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

  return true;
}

void DynfuEval::startMainLoop()
{
  image_capture_ptr_->start();

  sleep(2); // wait until RGB-D camera work stably

  // Register callbacks
  glutDisplayFunc(displayWrapper);
  glutKeyboardFunc(keyboardWrapper);
  glutMouseFunc(mouseWrapper);
  glutMotionFunc(motionWrapper);
  glutReshapeFunc(reshapeWrapper);
  glutTimerFunc(refresh_delay_, timerEventWrapper, 0);

  glutMainLoop();
}

void DynfuEval::display()
{
  ColorFrameHandle color_frame_handle;
  DepthFrameHandle depth_frame_handle;

  if (!pause_)
  {
    int fetch_delay = (image_capture_ptr_->isStopped()) ? 1 : 500;

    if (image_capture_ptr_->fetchFrame(&color_frame_handle, &depth_frame_handle, fetch_delay))
    {
      preprocess(*depth_frame_handle);

      gpu::bindImageTextures(intrinsics_.width,
                             intrinsics_.height,
                             vertex_image_,
                             normal_image_,
                             depth_scaled_);

      bool success = true;

      if (first_frame_)
      {
        reset();
        success = initDeformVoxelHierarchy();
        first_frame_ = false;
      }
      else
      {
        trackRigidMotion();
        frame_count_++;
        trackNonRigidMotion();
        fuseVolume();
      }

      if (success)
      {
        success = fetchSurface();
      }

      if (success)
      {
        warpSurface();
      }

      gpu::unbindImageTextures();

      if (success)
        success = updateDeformVoxelHierarchy();

      std::cout << '\r' << "DynfuEval: Processed frame [" << frame_count_ << "]" << std::flush;

      std::stringstream ss;
      ss << "CUGL DynamicFusion Demo" << "  voxel_block_size(" << dvh_ptr_->getVoxelBlockSize() << ")  node(" << dvh_ptr_->getNodeSize() << ")  mesh(" << mesh_vertex_size_ << ")";
      glutSetWindowTitle(ss.str().c_str());

      if (!success)
        pause_ = true;
    }
  }

  renderScene();
}

void DynfuEval::keyboard(unsigned char key, int x, int y)
{
  switch (key)
  {
    case (27):
      exit();
      break;
    case (32):
      pause_ = !pause_;
      image_capture_ptr_->togglePause();
      break;
    case ('c'):
      cam_view_ = !cam_view_;
      if (cam_view_)
      {
        rotate_ = make_float3(0.f, 0.f, 0.f);
        translate_ = make_float3(0.f, 0.f, 0.f);
      }
      break;
  }
}

void DynfuEval::mouse(int button, int state, int x, int y)
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

void DynfuEval::motion(int x, int y)
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
      translate_.y -= dy * 0.01f;
    }
    else if (mouse_buttons_ == 4)
    {
      translate_.z += dy * 0.01f;
    }
  }

  mouse_old_x_ = x;
  mouse_old_y_ = y;
}

void DynfuEval::reshape(int w, int h)
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, static_cast<float>(w) / static_cast<float>(h), 0.1, 10.0);

  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);
}

bool DynfuEval::initGL(int *argc, char **argv)
{
  // Create GL context
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(win_width_, win_height_);
  glutCreateWindow("CUGL DynamicFusion Demo");

  if (!isGLVersionSupported(2, 0))
  {
    std::cout << "Dynfu: Support for necessary OpenGL extensions missing." << std::endl;
    return false;
  }

  // default initialization
  glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
  glEnable(GL_DEPTH_TEST);

  float black[]     = { 0.f, 0.f, 0.f, 1.f };
  float white[]     = { 1.f, 1.f, 1.f, 1.f };
  float ambient[]   = { 0.1f, 0.1f, 0.1f, 1.f };
  float diffuse[]   = { 0.9f, 0.9f, 0.9f, 1.f };
  float light_pos[] = { 0.f, 0.f, -3.f, 0.f };

  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

  glLightfv(GL_LIGHT0, GL_AMBIENT, white);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
  glLightfv(GL_LIGHT0, GL_SPECULAR, white);
  glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

  glEnable(GL_LIGHT0);
  glEnable(GL_NORMALIZE);

  if (!compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code))
  {
    std::cout << "Dynfu: ASM shader compilation failed. " << std::endl;
    return false;
  }

  glutReportErrors();

  return true;
}

void DynfuEval::allocMemory(int image_width, int image_height)
{
  affines_.reserve(36000);

  depth_trunc_.alloc((size_t)image_width, (size_t)image_height);
  depth_scaled_.alloc((size_t)image_width, (size_t)image_height);
  depth_filtered_.alloc((size_t)image_width, (size_t)image_height);
  vertex_image_.alloc((size_t)image_width, (size_t)image_height);
  normal_image_.alloc((size_t)image_width, (size_t)image_height);

  // use OpenGL VBO
  mesh_vertex_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);
  mesh_normal_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);
  warped_mesh_vertex_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);
  warped_mesh_normal_array_.alloc((size_t)(3 * gpu::MAX_TRIANGLES_SIZE), false, true);

  surfel_vertex_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);
  surfel_normal_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);
  warped_surfel_vertex_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);
  warped_surfel_normal_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);

  active_surfel_flags_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);

  // use host buffer
  sum_buf_.alloc(27, true);
}

void DynfuEval::free()
{
  depth_trunc_.free();
  depth_scaled_.free();
  depth_filtered_.free();
  vertex_image_.free();
  normal_image_.free();

  mesh_vertex_array_.free();
  mesh_normal_array_.free();
  surfel_vertex_array_.free();
  surfel_normal_array_.free();

  warped_mesh_vertex_array_.free();
  warped_mesh_normal_array_.free();
  warped_surfel_vertex_array_.free();
  warped_surfel_normal_array_.free();

  active_surfel_flags_.free();

  sum_buf_.free();
}

void DynfuEval::reset()
{
  if (frame_count_)
  {
    std::cout << std::endl << RED << "Dynfu: Reset!" << RESET << std::endl;
  }

  // mode_ = PREVIEW;

  frame_count_ = 0;

  affines_.clear();

  affines_.push_back(affine_init_);

  dvh_ptr_->reset();
}

void DynfuEval::exit()
{
  std::cout << std::endl;

  glutDestroyWindow(glutGetWindow());
  image_capture_ptr_->stop();

  free();
}

bool DynfuEval::compileASMShader(GLenum program_type, const char *code)
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

void DynfuEval::preprocess(const cugl::DepthImage16u &depth_raw)
{
  float2 depth_thres = make_float2(configure_.depth_min,
                                   configure_.depth_max);

  gpu::truncateDepth(depth_thres,
                     intrinsics_,
                     depth_raw,
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
}

void DynfuEval::trackRigidMotion()
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
    dvh_ptr_->formIcpGaussNewtonOpt(T_curr_device,
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

void DynfuEval::trackNonRigidMotion()
{
  Eigen::Affine3f T_curr = affines_[frame_count_];

  cugl::Affine3d T_curr_device(T_curr.data());

  size_t surfel_vertex_size = mesh_vertex_size_ / 3;

  gpu::bindMeshTextures(surfel_vertex_array_,
                        surfel_normal_array_);

  bool success = dvh_ptr_->initDeformEnergyTerms(T_curr_device,
                                                 surfel_vertex_size,
                                                 energy_init_);

  if (!success)
  {
    std::cout << std::endl << RED << "DynfuEval: deform energy terms initialization failed!" << RESET << std::endl;
  }
  else
  {
    float energy_last = energy_init_;
    float energy_curr = energy_init_;

    // Solve non-rigid motion tracking based on the Gauss-Newton method
    for (int iter = 0; iter < configure_.icp_iters; iter++)
    {
      dvh_ptr_->formDetGaussNewtonOpt();

      float damping_old = 0.f;
      float damping_new = configure_.lm_damping;

      for (int lm_iter = 0; lm_iter < configure_.lm_iters; lm_iter++)
      {
        bool valid_result = dvh_ptr_->solveDetGaussNewtonOpt(damping_new - damping_old); // solve with damping factor

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

          dvh_ptr_->initTransformSearch();

          while (step_size > step_size_end)
          {
            dvh_ptr_->runTransformSearchOnce(step_size);

            dvh_ptr_->updateDeformEnergyTerms(T_curr_device,
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

bool DynfuEval::initDeformVoxelHierarchy()
{
  Eigen::Affine3f T_curr = affines_[frame_count_];

  Eigen::Matrix<float, 3, 3> R_curr = T_curr.linear();
  Eigen::Vector3f t_curr = T_curr.translation();

  Eigen::Matrix<float, 3, 3> R_curr_inv = R_curr.inverse();
  Eigen::Vector3f t_curr_inv = -(R_curr_inv * t_curr);

  Eigen::Affine3f T_curr_inv;
  T_curr_inv.linear() = R_curr_inv;
  T_curr_inv.translation() = t_curr_inv;

  cugl::Affine3d T_curr_device(T_curr.data());
  cugl::Affine3d T_curr_inv_device(T_curr_inv.data());

  bool success = dvh_ptr_->init(T_curr_device,
                                T_curr_inv_device);

  if (!success)
    std::cout << std::endl << RED << "DynfuEval: Deformable Voxel Hierarchy initilization failed!" << RESET << std::endl;

  return success;
}

void DynfuEval::fuseVolume()
{
  Eigen::Affine3f T_curr = affines_[frame_count_];
  cugl::Affine3d T_curr_device(T_curr.data());

  dvh_ptr_->fuseVolume(T_curr_device);
}

bool DynfuEval::fetchSurface()
{
  mesh_vertex_size_ = dvh_ptr_->computeIsoSurface(mesh_vertex_array_,
                                                  mesh_normal_array_,
                                                  surfel_vertex_array_,
                                                  surfel_normal_array_);

  return (mesh_vertex_size_ != 0);
}

void DynfuEval::warpSurface()
{
  Eigen::Affine3f T_curr = affines_[frame_count_];
  cugl::Affine3d T_curr_device(T_curr.data());

  dvh_ptr_->warpSurface(T_curr_device,
                        mesh_vertex_size_,
                        mesh_vertex_array_,
                        mesh_normal_array_,
                        warped_mesh_vertex_array_,
                        warped_mesh_normal_array_,
                        warped_surfel_vertex_array_,
                        warped_surfel_normal_array_,
                        active_surfel_flags_);
}

bool DynfuEval::updateDeformVoxelHierarchy()
{
  bool success = dvh_ptr_->update(mesh_vertex_size_ / 3,
                                  surfel_vertex_array_,
                                  active_surfel_flags_);

  return success;
}

void DynfuEval::renderScene()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // set view matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glRotatef(180.0f, 1.f, 0.f, 0.f);

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glEnable(GL_LIGHTING);

  // use camera view
  if (cam_view_)
    affine_view_ = affines_[frame_count_];

  glMultMatrixf(affine_view_.data());

  const float offset = configure_.volume_length / 2.0f;
  glTranslatef(offset, offset, offset);

  glTranslatef(translate_.x, translate_.y, translate_.z);
  glRotatef(rotate_.x, 1.0, 0.0, 0.0);
  glRotatef(rotate_.y, 0.0, -1.0, 0.0);

  glBindBuffer(GL_ARRAY_BUFFER, warped_mesh_vertex_array_.getVbo());
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, warped_mesh_normal_array_.getVbo());
  glNormalPointer(GL_FLOAT, sizeof(float)*4, 0);
  glEnableClientState(GL_NORMAL_ARRAY);

  glColor3f(1.0, 0.0, 0.0);
  glDrawArrays(GL_TRIANGLES, 0, mesh_vertex_size_);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // draw the TSDF volume
  glColor3f(0.1, 1.0, 1.0);
  glutWireCube(configure_.volume_length);

  // draw the camera frustum
  if (!cam_view_)
  {
    const float offset = configure_.volume_length / 2.0f;
    Eigen::Affine3f T_curr = affines_[frame_count_];
    Eigen::Matrix<float, 3, 3> R_curr = T_curr.linear();
    Eigen::Vector3f t_curr = T_curr.translation();

    Eigen::Vector3f cam_pos = Eigen::Vector3f(-offset, -offset, -offset) - R_curr.inverse() * t_curr;
    glTranslatef(cam_pos(0), cam_pos(1), cam_pos(2));
    glColor3f(1.0, 1.0, 0.0);
    glutSolidSphere(0.04, 20, 10);
  }

  glDisable(GL_LIGHTING);

  glutSwapBuffers();
  glutReportErrors();
}
} // namespace

int main(int argc, char **argv)
{
  dynfu::DynfuEval app(33, 1280, 960);

  if(app.init(argc, argv))
  {
    app.startMainLoop();
  }

  return 0;
}
