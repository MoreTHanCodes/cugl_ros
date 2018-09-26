#ifndef __CUGL_APPS_DYNFU_STREAM_CAPTURE_H__
#define __CUGL_APPS_DYFFU_STREAM_CAPTURE_H__

#include <atomic>
#include <queue>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include <librealsense/rs.hpp>

#include "cuda_internal.h"

namespace dynfu
{
/* Stream capture class of DynamicFusion
 *
 * Our implementation of DynamicFusion consists of two components:
 * the front end and the back end.
 * These two components run asynchronously and exchange data via
 * a concurrent queue.
 * The front end fetches synchronized RGB-D stream (undistorted)
 * from an Intel RealSense sensor.
 *
 */
class StreamCapture
{
 public:
  typedef std::shared_ptr<cugl::ColorImage8u> ColorFrameHandle;
  typedef std::shared_ptr<cugl::DepthImage16u> DepthFrameHandle;
  typedef std::chrono::duration<int, std::milli> MilliSeconds;
  typedef std::shared_ptr<StreamCapture> Ptr;
  
  StreamCapture(size_t max_queue_size = 30);
  ~StreamCapture();
  
  bool init();
  
  void start();
  
  void stop();

  void saveStream();
  
  /* Fetch the RGB-D frame pair
   * The calling will be blocked untill new frames are available.
   */
  bool fetchFrame(double *time_stamp_ptr, ColorFrameHandle *color_frame_hptr, DepthFrameHandle *depth_frame_hptr, int delay = 500);
  
  void getIntrinsics(gpu::Intrinsics &intrin) const
  {
    intrin.width = intrin_.width;
    intrin.height = intrin_.height;
    intrin.cx = intrin_.ppx;
    intrin.cy = intrin_.ppy;
    intrin.fx = intrin_.fx;
    intrin.fy = intrin_.fy;
    intrin.depth_scale = depth_scale_;
  }
  
  float getDepthScale() const { return depth_scale_; }

 private:
  void capture();

  void free();

  std::atomic<bool> stop_;
  std::atomic<bool> save_stream_;
  int frame_count_;
  std::thread thread_;

  rs::intrinsics intrin_;
  float depth_scale_;

  size_t max_queue_size_;
  std::mutex mutex_;
  std::condition_variable cond_not_empty_;
  std::queue<double> time_queue_;
  std::queue<ColorFrameHandle> color_queue_;
  std::queue<DepthFrameHandle> depth_queue_;
};

} // namespace dynfu

#endif /* __CUGL_APPS_DYNFU_STREAM_CAPTURE_H__ */
