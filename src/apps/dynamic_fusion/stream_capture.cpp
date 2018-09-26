#include <cugl_ros/apps/dynamic_fusion/stream_capture.h>

namespace dynfu
{
StreamCapture::StreamCapture(size_t max_queue_size)
    : stop_(false),
      save_stream_(false),
      frame_count_(0),
      thread_(),
      depth_scale_(0.f),
      max_queue_size_(max_queue_size),
      mutex_(),
      cond_not_empty_(),
      time_queue_(),
      color_queue_(),
      depth_queue_()
{
}

StreamCapture::~StreamCapture()
{
  free();
}

bool StreamCapture::init()
{
  rs::log_to_console(rs::log_severity::warn);

  rs::context ctx;
  std::cout << "StreamCapture: There are " << ctx.get_device_count() << " connected RealSense device." << std::endl;

  if(ctx.get_device_count() == 0)
  {
    std::cout << "StreamCapture: Init Failed!" << std::endl;
    return false;
  }

  rs::device &dev = *ctx.get_device(0);
  std::cout << "StreamCapture: Using device 0, an " << dev.get_name() << std::endl;

  dev.enable_stream(rs::stream::depth, rs::preset::best_quality);
  dev.enable_stream(rs::stream::color, rs::preset::best_quality);

  intrin_ = dev.get_stream_intrinsics(rs::stream::depth_aligned_to_rectified_color);
  depth_scale_ = dev.get_depth_scale();

  return true;
}

void StreamCapture::start()
{
  stop_ = false;

  thread_ = std::thread(&StreamCapture::capture, this);
}

void StreamCapture::stop()
{
  stop_ = true;

  if(thread_.joinable())
  {
    thread_.join();
  }
}

void StreamCapture::saveStream()
{
  save_stream_ = true;
}

bool StreamCapture::fetchFrame(double *time_stamp_ptr, ColorFrameHandle *color_frame_hptr, DepthFrameHandle *depth_frame_hptr, int delay)
{
  std::unique_lock<std::mutex> lock(mutex_);
  MilliSeconds milli_seconds(delay);
  const auto ready = [this]() { return !time_queue_.empty(); };

  if(!ready() && !cond_not_empty_.wait_for(lock, milli_seconds, ready))
  {
    return false;
  }

  auto time_stamp = std::move(time_queue_.front());
  auto color_frame_handle = std::move(color_queue_.front());
  auto depth_frame_handle = std::move(depth_queue_.front());

  time_queue_.pop();
  color_queue_.pop();
  depth_queue_.pop();

  *time_stamp_ptr = std::move(time_stamp);
  *color_frame_hptr = std::move(color_frame_handle);
  *depth_frame_hptr = std::move(depth_frame_handle);

  return true;
}

void StreamCapture::capture()
{
  rs::log_to_console(rs::log_severity::warn);

  rs::context ctx;
  rs::device &dev = *ctx.get_device(0);
  dev.enable_stream(rs::stream::depth, rs::preset::best_quality);
  dev.enable_stream(rs::stream::color, rs::preset::best_quality);

  intrin_ = dev.get_stream_intrinsics(rs::stream::rectified_color);
  depth_scale_ = dev.get_depth_scale();

  std::cout << "StreamCapture: Try to start realsense device ..." << std::endl;
  dev.start();
  std::cout << "StreamCapture: Done!" << std::endl;
  std::cout << "Color Format: " << dev.get_stream_format(rs::stream::rectified_color) << std::endl;
  std::cout << "Depth Format: " << dev.get_stream_format(rs::stream::depth_aligned_to_rectified_color) << std::endl;

  while (!stop_)
  {
    dev.wait_for_frames();

    const unsigned char  *color_data = (const unsigned char *)dev.get_frame_data(rs::stream::rectified_color);
    const unsigned short *depth_data = (const unsigned short *)dev.get_frame_data(rs::stream::depth_aligned_to_rectified_color);

    if (save_stream_)
    {
      std::stringstream ss_color;
      ss_color << "./frame_" << frame_count_ << ".color.png";
      std::stringstream ss_depth;
      ss_depth << "./frame_" << frame_count_ << ".depth.png";
      cv::Mat cv_color_frame(intrin_.height, intrin_.width, CV_8UC3, (void *)color_data, intrin_.width * sizeof(unsigned char) * 3);
      cv::Mat cv_depth_frame(intrin_.height, intrin_.width, CV_16UC1, (void *)depth_data, intrin_.width * sizeof(unsigned short));
      cv::cvtColor(cv_color_frame, cv_color_frame, cv::COLOR_RGB2BGR);
      cv::imwrite(ss_color.str(), cv_color_frame);
      cv::imwrite(ss_depth.str(), cv_depth_frame);

      std::cout << '\r' << "StreamCapture: Saved image frame [" << frame_count_ << "]" << std::flush;

      frame_count_++;
    }

    ColorFrameHandle color_frame_handle = std::make_shared<cugl::ColorImage8u>();
    color_frame_handle->alloc((size_t)(3 * intrin_.width), (size_t)intrin_.height);
    color_frame_handle->upload(color_data);

    DepthFrameHandle depth_frame_handle = std::make_shared<cugl::DepthImage16u>();
    depth_frame_handle->alloc((size_t)intrin_.width, (size_t)intrin_.height);
    depth_frame_handle->upload(depth_data);

    double time_stamp = dev.get_frame_timestamp(rs::stream::rectified_color);

    std::unique_lock<std::mutex> lock(mutex_);

    if (time_queue_.size() == max_queue_size_)
    {
      time_queue_.pop();
      color_queue_.pop();
      depth_queue_.pop();
    }

    time_queue_.push(std::move(time_stamp));
    color_queue_.push(std::move(color_frame_handle));
    depth_queue_.push(std::move(depth_frame_handle));

    lock.unlock();

    cond_not_empty_.notify_one();
  }

  if (dev.is_streaming())
  {
    std::cout << std::endl << "StreamCapture: Try to stop realsense device ..." << std::endl;
    dev.stop();
    std::cout << "StreamCapture: Done!" << std::endl;
  }
}

void StreamCapture::free()
{
  std::unique_lock<std::mutex> lock(mutex_);
  while (time_queue_.size() > 0)
  {
    time_queue_.pop();
    color_queue_.pop();
    depth_queue_.pop();
  }
}
} // namespace
