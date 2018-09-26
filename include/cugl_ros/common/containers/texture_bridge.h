#ifndef __CUGL_COMMON_CONTAINERS_TEXTURE_BRIDGE_H__
#define __CUGL_COMMON_CONTAINERS_TEXTURE_BRIDGE_H__

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cugl_ros/common/helper_gl.h>
#include <cugl_ros/common/helper_cuda.h>
#include <cugl_ros/common/helper_cuda_gl.h>

namespace cugl
{
/*
 * Wrapper class for interoperation between texture (object) in OpenGL and CUDA
 *
 * We use a CUDA array as the entry of the texture in CUDA, and bind
 * it with a texture object in OpenGL.
 *
 */
class TextureBridge
{
 public:
  enum Mode
  {
    WRITE_ONLY,
    READ_ONLY
  };

  TextureBridge();
  ~TextureBridge();

  cudaArray *getCudaArrayPtr()
  {
    return cuda_array_;
  }

  const cudaArray *getCudaArrayPtr() const
  {
    return static_cast<const cudaArray *>(cuda_array_);
  }

  GLuint getTexture() const
  {
    return texture_;
  }

  size_t getWidth() const
  {
    return width_;
  }

  size_t getHeight() const
  {
    return height_;
  }

  // allocate and free
  void create(size_t width, size_t height, GLuint texture, Mode mode);
  void free();

  void map();
  void unmap();


 private:
  size_t width_, height_;
  GLuint texture_;
  struct cudaGraphicsResource *cuda_texture_resource_;
  cudaArray *cuda_array_;
};
} // namespace

#endif /* __CUGL_COMMON_CONTAINERS_TEXTURE_BRIDGE_H__ */
