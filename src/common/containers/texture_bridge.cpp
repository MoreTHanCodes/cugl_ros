#include <cugl_ros/common/containers/texture_bridge.h>

namespace cugl
{
TextureBridge::TextureBridge()
    : width_(0),
      height_(0),
      texture_(0),
      cuda_texture_resource_(NULL),
      cuda_array_(0)
{
}

TextureBridge::~TextureBridge()
{
  free();
}

void TextureBridge::create(size_t width, size_t height, GLuint texture, Mode mode)
{
  width_ = width;
  height_ = height;
  texture_ = texture;

  switch(mode)
  {
    case WRITE_ONLY:
      checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_texture_resource_, texture_,
                                                  GL_TEXTURE_2D,
                                                  cudaGraphicsMapFlagsWriteDiscard));
      break;

    case READ_ONLY:
      checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_texture_resource_, texture_,
                                                  GL_TEXTURE_2D,
                                                  cudaGraphicsMapFlagsReadOnly));
      break;
  }
}

void TextureBridge::free()
{
  if(texture_)
  {
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_texture_resource_));
    texture_ = 0;
  }
}

void TextureBridge::map()
{
  if(texture_)
  {
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_texture_resource_, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_array_,
                                                          cuda_texture_resource_,
                                                          0, 0));
  }
}

void TextureBridge::unmap()
{
  if(texture_)
  {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_texture_resource_, 0));
    cuda_array_ = 0;
  }
}
} // namespace
