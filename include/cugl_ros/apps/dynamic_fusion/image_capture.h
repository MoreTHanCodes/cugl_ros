#ifndef __CUGL_APPS_DYNFU_IMAGE_CAPTURE_H__
#define __CUGL_APPS_DYNFU_IMAGE_CAPTURE_H__

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
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cuda_internal.h"

namespace dynfu
{
/* Image Capture Wrapper Class
 */

class ImageCapture
{
 public:
  typedef std::shared_ptr<cugl::ColorImage8u> ColorFrameHandle;
  typedef std::shared_ptr<cugl::DepthImage16u> DepthFrameHandle;
  typedef std::chrono::duration<int, std::milli> MilliSeconds;
  typedef std::shared_ptr<ImageCapture> Ptr;

  ImageCapture(const std::string &folder, size_t max_queue_size = 0, int width = 640, int height = 480);
  ~ImageCapture();

  bool init();

  void start();

  void stop();

  bool isStopped() const;

  void togglePause();

  bool fetchFrame(ColorFrameHandle *color_frame_hptr, DepthFrameHandle *depth_frame_hptr, int delay = 500);

 private:
  void capture();

  void free();

  bool readFile(const std::string &file, std::vector< std::pair<double, std::string> > &output);

  std::atomic<bool> pause_;
  std::atomic<bool> stop_;
  int frame_count_;
  std::thread thread_;

  std::string folder_;

  std::vector< std::pair<double, std::string> > color_stamps_and_filenames_;
  std::vector< std::pair<double, std::string> > depth_stamps_and_filenames_;

  size_t max_queue_size_;
  int width_, height_;
  std::mutex mutex_;
  std::condition_variable cond_not_empty_;
  std::queue<ColorFrameHandle> color_queue_;
  std::queue<DepthFrameHandle> depth_queue_;
};

} // namespace dynfu

#endif /* __CUGL_APPS_DYNFU_IMAGE_CAPTURE_H__ */
