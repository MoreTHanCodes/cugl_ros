#ifndef __CUGL_COMMON_CONTAINERS_DEVICE_ARRAY_HPP__
#define __CUGL_COMMON_CONTAINERS_DEVICE_ARRAY_HPP__

#include <stdlib.h>
#include <stdio.h>
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cugl_ros/common/helper_gl.h>
#include <cugl_ros/common/helper_cuda.h>
#include <cugl_ros/common/helper_cuda_gl.h>

#include <cugl_ros/common/containers/device_array_handle.h>

namespace cugl
{
/*
 * Class to represent an array in GPU memory.
 * It has an interoperability with OpenGL VBO and CPU memory.
 */
template <typename T>
class DeviceArray
{
 public:
  enum Direction
  {
    HOST_TO_DEVICE,
    DEVICE_TO_HOST
  };

  DeviceArray();
  ~DeviceArray();

  T *getDevicePtr()
  {
    return dptr_[current_read_];
  }
  
  const T *getDevicePtr() const
  {
    return reinterpret_cast<const T *>(dptr_[current_read_]);
  }
  
  T *getDeviceWritePtr()
  {
    return dptr_[current_write_];
  }
  
  const T *getDeviceWritePtr() const
  {
    return reinterpret_cast<const T *>(dptr_[current_write_]);
  }
  
  GLuint getVbo() const
  {
    return vbo_[current_read_];
  }
  
  GLuint getWriteVbo() const
  {
    return vbo_[current_write_];
  }
  
  T *getHostPtr()
  {
    return hptr_;
  }
  
  const T *getHostPtr() const
  {
    return reinterpret_cast<const T *>(hptr_);
  }

  DeviceArrayHandle<T> getHandle()
  {
    DeviceArrayHandle<T> handle(this->getDevicePtr());
    return handle;
  }

  DeviceArrayHandle<T> getHandle(uint offset)
  {
    DeviceArrayHandle<T> handle(this->getDevicePtr() + offset);
    return handle;
  }

  const DeviceArrayHandle<T> getHandle() const
  {
    const DeviceArrayHandle<T> handle(const_cast<T *>(this->getDevicePtr()));
    return handle;
  }

  const DeviceArrayHandle<T> getHandle(uint offset) const
  {
    const DeviceArrayHandle<T> handle(const_cast<T *>(this->getDevicePtr() + offset));
    return handle;
  }

  DeviceArrayHandle<T> getWriteHandle()
  {
    DeviceArrayHandle<T> handle(this->getDeviceWritePtr());
    return handle;
  }

  DeviceArrayHandle<T> getWriteHandle(uint offset)
  {
    DeviceArrayHandle<T> handle(this->getDeviceWritePtr() + offset);
    return handle;
  }

  const DeviceArrayHandle<T> getWriteHandle() const
  {
    const DeviceArrayHandle<T> handle(const_cast<T *>(this->getDeviceWritePtr()));
    return handle;
  }

  const DeviceArrayHandle<T> getWriteHandle(uint offset) const
  {
    const DeviceArrayHandle<T> handle(const_cast<T *>(this->getDeviceWritePtr() + offset));
    return handle;
  }
  
  size_t getSize() const
  {
    return size_;
  }
  
  // allocate and free
  void alloc(size_t size, bool use_host = false, bool use_vbo = false, bool use_double_buffer = false, bool use_element_array = false);
  void free();
  
  // swap buffers for double buffering
  void swap();
  
  // when using vbo, must map before getting device ptr
  void map();
  void unmap();
  
  void copy(Direction dir, uint start = 0, uint count = 0);
  
  void upload(const T *hptr, uint start = 0, uint count = 0);
  void download(T *hptr, uint start = 0, uint count = 0);

 private:
  GLuint createVbo(size_t size, bool use_element_array);
  
  void allocDevice();
  void allocVbo(bool use_element_array);
  void allocHost();
  
  void freeDevice();
  void freeVbo();
  void freeHost();
  
  size_t size_;
  T *dptr_[2];
  GLuint vbo_[2];
  struct cudaGraphicsResource *cuda_vbo_resource_[2]; // handles OpenGL-CUDA exchange
  
  T *hptr_;
  
  bool use_host_;
  bool use_vbo_;
  bool use_double_buffer_;
  uint current_read_, current_write_;
};

/*
* Class to represent a 2D texture in GPU linear memory.
* Please bind the linear memory with a texture reference before a CUDA kernel call.
*/
template <typename T>
class DeviceArray2D
{
 public:
  enum Direction
  {
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
  };

  DeviceArray2D();
  ~DeviceArray2D();

  T *getDevicePtr()
  {
    return dptr_[current_read_];
  }

  const T *getDevicePtr() const
  {
    return reinterpret_cast<const T *>(dptr_[current_read_]);
  }

  T *getDeviceWritePtr()
  {
    return dptr_[current_write_];
  }

  const T *getDeviceWritePtr() const
  {
    return reinterpret_cast<const T *>(dptr_[current_write_]);
  }

  T *getHostPtr()
  {
    return hptr_;
  }

  const T *getHostPtr() const
  {
    return reinterpret_cast<const T *>(hptr_);
  }

  DeviceArrayHandle2D<T> getHandle()
  {
    DeviceArrayHandle2D<T> handle(this->getDevicePtr(), this->getPitch());
    return handle;
  }

  DeviceArrayHandle2D<T> getHandle(uint offset_w, uint offset_h)
  {
    T *row_entry = reinterpret_cast<T *>(reinterpret_cast<char *>(this->getDevicePtr()) + offset_h * this->getPitch());

    DeviceArrayHandle2D<T> handle(row_entry + offset_w, this->getPitch());
    return handle;
  }

  const DeviceArrayHandle2D<T> getHandle() const
  {
    const DeviceArrayHandle2D<T> handle(const_cast<T *>(this->getDevicePtr()), this->getPitch());
    return handle;
  }

  const DeviceArrayHandle2D<T> getHandle(uint offset_w, uint offset_h) const
  {
    T *row_entry = reinterpret_cast<T *>(reinterpret_cast<char *>(this->getDevicePtr()) + offset_h * this->getPitch());

    const DeviceArrayHandle2D<T> handle(const_cast<T *>(row_entry + offset_w), this->getPitch());
    return handle;
  }

  size_t getWidth() const
  {
    return width_;
  }

  size_t getHeight() const
  {
    return height_;
  }

  size_t getPitch() const
  {
    return pitch_;
  }

  // allocate and free
  void alloc(size_t width, size_t height, bool use_host = false, bool use_double_buffer = false);
  void free();

  // swap buffers for double buffering
  void swap();

  void copy(Direction dir);

  void upload(const T *hptr);

 private:
  void allocDevice();
  void allocHost();

  void freeDevice();
  void freeHost();

  size_t width_, height_; // width and height in element units
  size_t pitch_; // allocated width in bytes, important for element accessing in CUDA

  T *dptr_[2];
  T *hptr_;

  bool use_host_;
  bool use_double_buffer_;
  uint current_read_, current_write_;
};
} // namespace

#include <cugl_ros/common/containers/impl/device_array.hpp>

#endif /* __CUGL_COMMON_CONTAINERS_DEVICE_ARRAY_HPP__ */
