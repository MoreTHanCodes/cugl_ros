#ifndef __CUGL_APPS_DYNFU_CONFIGURE_H__
#define __CUGL_APPS_DYNFU_CONFIGURE_H__

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>

namespace dynfu
{
class DynfuConfigure
{
 public:
  float rx, ry, rz;
  float tx, ty, tz;
  float trunc_dist;
  float volume_length;
  int voxel_block_inner_dim_shift;
  int voxel_block_dim_shift;
  int node_block_dim_shift_base;
  int roi_start_x;
  int roi_start_y;
  int roi_end_x;
  int roi_end_y;
  float depth_min;
  float depth_max;
  float h_min;
  float h_max;
  float s_min;
  float s_max;
  float dist_thres;
  float angle_thres;
  float view_angle_thres;
  int width, height;
  float cx, cy, fx, fy;
  float depth_scale;
  float w_fit;
  float w_fit_huber;
  float w_reg;
  float w_reg_huber;
  int icp_iters;
  int lm_iters;
  float lm_damping;
  float lm_alpha;
  float lm_beta;
  int draw_mesh_win1;
  int draw_tex_win0;

  DynfuConfigure()
      : rx(0.f),
        ry(0.f),
        rz(0.f),
        tx(-0.5f),
        ty(-0.5f),
        tz(0.5f),
        trunc_dist(0.02f),
        volume_length(1.0f),
        voxel_block_inner_dim_shift(3),
        voxel_block_dim_shift(6),
        node_block_dim_shift_base(5),
        depth_min(0.5f),
        depth_max(2.0f),
        h_min(160.f),
        h_max(210.f),
        s_min(0.4f),
        s_max(1.f),
        dist_thres(0.008f),
        angle_thres(0.96f),
        view_angle_thres(1.41f),
        width(640),
        height(480),
        cx(0.0f),
        cy(0.0f),
        fx(0.0f),
        fy(0.0f),
        depth_scale(1000.0f),
        w_fit(1.f),
        w_fit_huber(999.f),
        w_reg(0.8f),
        w_reg_huber(0.008f),
        icp_iters(4),
        lm_iters(4),
        lm_damping(5.f),
        lm_alpha(0.5f),
        lm_beta(2.5f),
        draw_mesh_win1(0),
        draw_tex_win0(0)
  {}

  void write(cv::FileStorage &fs) const
  {
    fs << "{"
       << "rx" << rx
       << "ry" << ry
       << "rz" << rz
       << "tx" << tx
       << "ty" << ty
       << "tz" << tz
       << "trunc_dist" << trunc_dist
       << "volume_length" << volume_length
       << "voxel_block_inner_dim_shift" << voxel_block_inner_dim_shift
       << "voxel_block_dim_shift" << voxel_block_dim_shift
       << "node_block_dim_shift_base" << node_block_dim_shift_base
       << "depth_min" << depth_min
       << "depth_max" << depth_max
       << "h_min" << h_min
       << "h_max" << h_max
       << "s_min" << s_min
       << "s_max" << s_max
       << "dist_thres" << dist_thres
       << "angle_thres" << angle_thres
       << "view_angle_thres" << view_angle_thres
       << "width" << width
       << "height" << height
       << "cx" << cx
       << "cy" << cy
       << "fx" << fx
       << "fy" << fy
       << "depth_scale" << depth_scale
       << "w_fit" << w_fit
       << "w_fit_huber" << w_fit_huber
       << "w_reg" << w_reg
       << "w_reg_huber" << w_reg_huber
       << "icp_iters" << icp_iters
       << "lm_iters" << lm_iters
       << "lm_damping" << lm_damping
       << "lm_alpha" << lm_alpha
       << "lm_beta" << lm_beta
       << "draw_mesh_win1" << draw_mesh_win1
       << "draw_tex_win0" << draw_tex_win0
       << "}";
  }

  void read(const cv::FileNode &node)
  {
    rx = static_cast<float>(node["rx"]);
    ry = static_cast<float>(node["ry"]);
    rz = static_cast<float>(node["rz"]);
    tx = static_cast<float>(node["tx"]);
    ty = static_cast<float>(node["ty"]);
    tz = static_cast<float>(node["tz"]);
    trunc_dist = static_cast<float>(node["trunc_dist"]);
    volume_length = static_cast<float>(node["volume_length"]);
    voxel_block_inner_dim_shift = static_cast<int>(node["voxel_block_inner_dim_shift"]);
    voxel_block_dim_shift = static_cast<int>(node["voxel_block_dim_shift"]);
    node_block_dim_shift_base = static_cast<int>(node["node_block_dim_shift_base"]);
    depth_min = static_cast<float>(node["depth_min"]);
    depth_max = static_cast<float>(node["depth_max"]);
    h_min = static_cast<float>(node["h_min"]);
    h_max = static_cast<float>(node["h_max"]);
    s_min = static_cast<float>(node["s_min"]);
    s_max = static_cast<float>(node["s_max"]);
    dist_thres = static_cast<float>(node["dist_thres"]);
    angle_thres = static_cast<float>(node["angle_thres"]);
    view_angle_thres = static_cast<float>(node["view_angle_thres"]);
    width = static_cast<int>(node["width"]);
    height = static_cast<int>(node["height"]);
    cx = static_cast<float>(node["cx"]);
    cy = static_cast<float>(node["cy"]);
    fx = static_cast<float>(node["fx"]);
    fy = static_cast<float>(node["fy"]);
    depth_scale = static_cast<float>(node["depth_scale"]);
    w_fit = static_cast<float>(node["w_fit"]);
    w_fit_huber = static_cast<float>(node["w_fit_huber"]);
    w_reg = static_cast<float>(node["w_reg"]);
    w_reg_huber = static_cast<float>(node["w_reg_huber"]);
    icp_iters = static_cast<int>(node["icp_iters"]);
    lm_iters = static_cast<int>(node["lm_iters"]);
    lm_damping = static_cast<float>(node["lm_damping"]);
    lm_alpha = static_cast<float>(node["lm_alpha"]);
    lm_beta = static_cast<float>(node["lm_beta"]);
    draw_mesh_win1 = static_cast<int>(node["draw_mesh_win1"]);
    draw_tex_win0 = static_cast<int>(node["draw_tex_win0"]);
  }
};

static void write(cv::FileStorage &fs, const std::string &, const DynfuConfigure &configure)
{
  configure.write(fs);
}

static void read(const cv::FileNode &node, DynfuConfigure &configure, const DynfuConfigure &default_configure = DynfuConfigure())
{
  if (node.empty())
  {
    configure = default_configure;
    std::cout << "DynfuConfigure: Reading failed, parameters have been set as default values!" << std::endl;
  }
  else
  {
    configure.read(node);
    std::cout << "DynfuConfigure: Reading successed, parameters have been set as specific values!" << std::endl;
  }
}

static std::ostream &operator<<(std::ostream &out, const DynfuConfigure &configure)
{
  out << "********** Dynfu Configuration **********" << std::endl;
  out << "rx = " << configure.rx << ", ry = " << configure.ry << ", rz = " << configure.rz << std::endl;
  out << "tx = " << configure.tx << ", ty = " << configure.ty << ", tz = " << configure.tz << std::endl;
  out << "trunc_dist = " << configure.trunc_dist << ", volume_length = " << configure.volume_length << std::endl;
  out << "voxel_block_inner_dim_shift = " << configure.voxel_block_inner_dim_shift << ", voxel_block_dim_shift = " << configure.voxel_block_dim_shift << ", node_block_dim_shift_base = " << configure.node_block_dim_shift_base << std::endl;
  out << "depth_min = " << configure.depth_min << ", depth_max = " << configure.depth_max << std::endl;
  out << "h_min = " << configure.h_min << ", h_max = " << configure.h_max << std::endl;
  out << "s_min = " << configure.s_min << ", s_max = " << configure.s_max << std::endl;
  out << "dist_thres = " << configure.dist_thres << ", angle_thres = " << configure.angle_thres << ", view_angle_thres = " << configure.view_angle_thres << std::endl;
  out << "width = " << configure.width << ", height = " << configure.height << std::endl;
  out << "cx = " << configure.cx << ", cy = " << configure.cy << ", fx = " << configure.fx << ", fy = " << configure.fy << ", depth_scale = " << configure.depth_scale << std::endl;
  out << "w_fit = " << configure.w_fit << ", w_fit_huber = " << configure.w_fit_huber << ", w_reg = " << configure.w_reg << ", w_reg_huber = " << configure.w_reg_huber << std::endl;
  out << "icp_iters = " << configure.icp_iters << ", lm_iters = " << configure.lm_iters << ", lm_damping = " << configure.lm_damping << ", lm_alpha = " << configure.lm_alpha << ", lm_beta = " << configure.lm_beta << std::endl;
  out << "*****************************************";

  return out;
}
} // namespace

#endif /* __CUGL_APPS_DYNFU_CONFIGURE_H__ */
