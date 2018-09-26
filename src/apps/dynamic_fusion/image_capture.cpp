#include <cugl_ros/apps/dynamic_fusion/image_capture.h>

using namespace cv;

namespace dynfu
{
//////////////////////////////
// Public Methods
//////////////////////////////
ImageCapture::ImageCapture(const std::string &folder, size_t max_queue_size, int width, int height)
    : pause_(false),
      stop_(false),
      frame_count_(0),
      thread_(),
      folder_(folder),
      max_queue_size_(max_queue_size),
      width_(width),
      height_(height),
      mutex_(),
      cond_not_empty_(),
      color_queue_(),
      depth_queue_()
{
}

ImageCapture::~ImageCapture()
{
  free();
}

bool ImageCapture::init()
{
  if (folder_[folder_.size() - 1] != '/')
    folder_.push_back('/');

  std::cout << "ImageCapture: Read rgb-d images from folder " << folder_ << std::endl;

  std::string rgb_file = folder_ + "rgb.txt";
  std::string depth_file = folder_ + "depth.txt";

  bool success = true;

  success = readFile(rgb_file, color_stamps_and_filenames_);

  if (!success)
  {
    std::cout << "ImageCapture: Can't read file " << rgb_file << std::endl;
    return success;
  }

  success = readFile(depth_file, depth_stamps_and_filenames_);

  if (!success)
  {
    std::cout << "ImageCapture: Can't read file " << depth_file << std::endl;
    return success;
  }

  return success;
}

void ImageCapture::start()
{
  stop_ = false;

  thread_ = std::thread(&ImageCapture::capture, this);
}

void ImageCapture::stop()
{
  stop_ = true;

  if(thread_.joinable())
  {
    thread_.join();
  }
}

bool ImageCapture::isStopped() const
{
  return stop_;
}

void ImageCapture::togglePause()
{
  pause_ = !pause_;

  if (pause_)
  {
    std::cout << std::endl << "ImageCapture: Pause!" << std::endl;
  }
  else
  {
    std::cout << std::endl << "ImageCapture: Resume work!" << std::endl;
  }
}

bool ImageCapture::fetchFrame(ColorFrameHandle *color_frame_hptr, DepthFrameHandle *depth_frame_hptr, int delay)
{
  std::unique_lock<std::mutex> lock(mutex_);
  MilliSeconds milli_seconds(delay);
  const auto ready = [this]() { return !depth_queue_.empty(); };

  if(!ready() && !cond_not_empty_.wait_for(lock, milli_seconds, ready))
  {
    return false;
  }

  auto color_frame_handle = std::move(color_queue_.front());
  auto depth_frame_handle = std::move(depth_queue_.front());

  color_queue_.pop();
  depth_queue_.pop();

  *color_frame_hptr = std::move(color_frame_handle);
  *depth_frame_hptr = std::move(depth_frame_handle);

  return true;
}

//////////////////////////////
// Private Methods
//////////////////////////////
void ImageCapture::capture()
{
  int total = depth_stamps_and_filenames_.size();

  while (!stop_ && (frame_count_ < total-1))
  {
    if (!pause_)
    {
      // capture color image
      std::string color_file = folder_ + color_stamps_and_filenames_[frame_count_].second;

      Mat bgr_img = imread(color_file);

      if (bgr_img.empty())
      {
        std::cout << "ImageCapture: color image is empty [" << frame_count_ << "]" << std::endl;
        break;
      }

      Mat color_img;
      cvtColor(bgr_img, color_img, CV_BGR2RGB);

      const unsigned char *color_data = color_img.ptr<unsigned char>();

      ColorFrameHandle color_frame_handle = std::make_shared<cugl::ColorImage8u>();
      color_frame_handle->alloc((size_t)(3 * width_), (size_t)height_);
      color_frame_handle->upload(color_data);

      // capture depth image
      std::string depth_file = folder_ + depth_stamps_and_filenames_[frame_count_].second;

      Mat depth_img = imread(depth_file, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

      if (depth_img.empty())
      {
        std::cout << "ImageCapture: depth image is empty [" << frame_count_ << "]" << std::endl;
        break;
      }

      if (depth_img.elemSize() != sizeof(unsigned short))
      {
        std::cout << "ImageCapture: invalid depth image element size [" << frame_count_ << "]" << std::endl;
        break;
      }

      const unsigned short *depth_data = depth_img.ptr<unsigned short>();

      DepthFrameHandle depth_frame_handle = std::make_shared<cugl::DepthImage16u>();
      depth_frame_handle->alloc((size_t)width_, (size_t)height_);
      depth_frame_handle->upload(depth_data);

      // write to queue
      std::unique_lock<std::mutex> lock(mutex_);

      if ((max_queue_size_ != 0) && (depth_queue_.size() == max_queue_size_))
      {
        color_queue_.pop();
        depth_queue_.pop();
      }

      color_queue_.push(std::move(color_frame_handle));
      depth_queue_.push(std::move(depth_frame_handle));

      lock.unlock();

      cond_not_empty_.notify_one();

      frame_count_++;
    } // !pause_
  } // main loop

  if (frame_count_ == total - 1)
    stop_ = true;

  std::cout << std::endl << "ImageCapture: Stop reading image! [" << frame_count_ << "/" << (total-1) << "]" << std::endl;
}

void ImageCapture::free()
{
  std::unique_lock<std::mutex> lock(mutex_);
  while (depth_queue_.size() > 0)
  {
    color_queue_.pop();
    depth_queue_.pop();
  }
}

bool ImageCapture::readFile(const std::string &file, std::vector< std::pair<double, std::string> > &output)
{
  char buffer[4096];
  std::vector< std::pair<double, std::string> > tmp;

  std::ifstream iff(file.c_str());

  if (!iff)
    return false;

  // ignore three header lines
  iff.getline(buffer, sizeof(buffer));
  iff.getline(buffer, sizeof(buffer));
  iff.getline(buffer, sizeof(buffer));

  // each lne consists of the timestamp and the filename of the image
  while (!iff.eof())
  {
    double time;
    std::string name;
    iff >> time >> name;
    tmp.push_back(make_pair(time, name));
  }

  tmp.swap(output);

  return true;
}

} // namespace dynfu
