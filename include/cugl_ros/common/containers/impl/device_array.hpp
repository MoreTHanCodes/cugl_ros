#ifndef __CUGL_COMMON_CONTAINERS_DEVICE_ARRAY_IMPL_HPP__
#define __CUGL_COMMON_CONTAINERS_DEVICE_ARRAY_IMPL_HPP__

namespace cugl
{
/*
 * Class DeviceArray
 */
template <class T>
DeviceArray<T>::DeviceArray()
    : size_(0),
      hptr_(NULL),
      current_read_(0),
      current_write_(0)
{
  dptr_[0] = NULL;
  dptr_[1] = NULL;

  vbo_[0] = 0;
  vbo_[1] = 0;

  cuda_vbo_resource_[0] = NULL;
  cuda_vbo_resource_[1] = NULL;
}

template <class T>
DeviceArray<T>::~DeviceArray()
{
  free();
}

template <class T>
void DeviceArray<T>::alloc(size_t size, bool use_host, bool use_vbo, bool use_double_buffer, bool use_element_array)
{
  size_ = size;

  use_host_ = use_host;
  use_vbo_ = use_vbo;
  use_double_buffer_ = use_double_buffer;

  if(use_host_)
  {
    allocHost();
  }

  if(use_double_buffer_)
  {
    current_write_ = 1;
  }

  if(use_vbo_)
  {
    allocVbo(use_element_array);
  }
  else
  {
    allocDevice();
  }
}

template <class T>
void DeviceArray<T>::free()
{
  if(use_host_)
  {
    freeHost();
  }

  if(use_vbo_)
  {
    freeVbo();
  }
  else
  {
    freeDevice();
  }
}

template <class T>
void DeviceArray<T>::allocHost()
{
  hptr_ = (T *) new T [size_];
}

template <class T>
void DeviceArray<T>::freeHost()
{
  if(hptr_ != NULL)
  {
    delete [] hptr_;
    hptr_ = NULL;
  }
}

template <class T>
void DeviceArray<T>::allocDevice()
{
  checkCudaErrors(cudaMalloc((void **)&dptr_[0], size_ * sizeof(T)));

  if(use_double_buffer_)
  {
    checkCudaErrors(cudaMalloc((void **)&dptr_[1], size_ * sizeof(T)));
  }
}

template <class T>
void DeviceArray<T>::freeDevice()
{
  if(dptr_[0] != NULL)
  {
    checkCudaErrors(cudaFree(dptr_[0]));
    dptr_[0] = NULL;
  }

  if(dptr_[1] != NULL)
  {
    checkCudaErrors(cudaFree(dptr_[1]));
    dptr_[1] = NULL;
  }
}

template <class T>
GLuint DeviceArray<T>::createVbo(size_t size, bool use_element_array)
{
  GLuint vbo;
  glGenBuffers(1, &vbo);

  if(use_element_array)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  }
  else
  {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  return vbo;
}

template <class T>
void DeviceArray<T>::allocVbo(bool use_element_array)
{
  vbo_[0] = createVbo(size_ * sizeof(T),use_element_array);
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_[0], vbo_[0],
                                               cudaGraphicsMapFlagsWriteDiscard));

  if(use_double_buffer_)
  {
    vbo_[1] = createVbo(size_ * sizeof(T), use_element_array);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_[1], vbo_[1],
                                                 cudaGraphicsMapFlagsWriteDiscard));
  }
}

template <class T>
void DeviceArray<T>::freeVbo()
{
  if(vbo_[0] != 0)
  {
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_[0]));
    glDeleteBuffers(1, &vbo_[0]);
    vbo_[0] = 0;
  }

  if(vbo_[1] != 0)
  {
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_[1]));
    glDeleteBuffers(1, &vbo_[1]);
    vbo_[1] = 0;
  }
}

template <class T>
void DeviceArray<T>::swap()
{
  std::swap(current_read_, current_write_);
}

template <class T>
void DeviceArray<T>::map()
{
  if(vbo_[0])
  {
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_[0], 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&dptr_[0]),
                                                         &num_bytes,
                                                         cuda_vbo_resource_[0]));
  }

  if(use_double_buffer_ && vbo_[1])
  {
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_[1], 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&dptr_[1]),
                                                         &num_bytes,
                                                         cuda_vbo_resource_[1]));
  }
}

template <class T>
void DeviceArray<T>::unmap()
{
  if(vbo_[0])
  {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_[0], 0));
    dptr_[0] = 0;
  }

  if(use_double_buffer_ && vbo_[1])
  {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_[1], 0));
    dptr_[1] = 0;
  }
}

template <class T>
void DeviceArray<T>::copy(Direction dir, uint start, uint count)
{
  if(!use_host_)
  {
    allocHost();
    use_host_ = true;
  }

  if(count == 0)
  {
    count = (uint) size_;
  }

  map();

  switch(dir)
  {
    case HOST_TO_DEVICE:
      checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(dptr_[current_read_] + start), reinterpret_cast<void *>(hptr_ + start), count * sizeof(T), cudaMemcpyHostToDevice));
      break;

    case DEVICE_TO_HOST:
      checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(hptr_ + start), reinterpret_cast<void *>(dptr_[current_read_] + start), count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  unmap();
}

template <class T>
void DeviceArray<T>::upload(const T *hptr, uint start, uint count)
{
  if(count == 0)
  {
    count = static_cast<uint>(size_);
  }

  map();

  checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(dptr_[current_read_] + start), reinterpret_cast<const void *>(hptr + start), count * sizeof(T), cudaMemcpyHostToDevice));

  unmap();
}

template <class T>
void DeviceArray<T>::download(T *hptr, uint start, uint count)
{
  if(count == 0)
  {
    count = static_cast<uint>(size_);
  }

  map();

  checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(hptr + start), reinterpret_cast<void *>(dptr_[current_read_] + start), count * sizeof(T), cudaMemcpyDeviceToHost));

  unmap();
}

/*
 * Class DeviceArray2D
 */
template <class T>
DeviceArray2D<T>::DeviceArray2D()
    : width_(0),
      height_(0),
      pitch_(0),
      hptr_(NULL),
      current_read_(0),
      current_write_(0)
{
  dptr_[0] = NULL;
  dptr_[1] = NULL;
}

template <class T>
DeviceArray2D<T>::~DeviceArray2D()
{
  free();
}

template <class T>
void DeviceArray2D<T>::alloc(size_t width, size_t height, bool use_host, bool use_double_buffer)
{
  width_ = width;
  height_ = height;

  use_host_ = use_host;
  use_double_buffer_ = use_double_buffer;

  if(use_host_)
  {
    allocHost();
  }

  if(use_double_buffer_)
  {
    current_write_ = 1;
  }

  allocDevice();
}

template <class T>
void DeviceArray2D<T>::free()
{
  if(use_host_)
  {
    freeHost();
  }

  freeDevice();
}

template <class T>
void DeviceArray2D<T>::allocHost()
{
  hptr_ = (T *) new T [width_ * height_];
}

template <class T>
void DeviceArray2D<T>::freeHost()
{
  if(hptr_ != NULL)
  {
    delete [] hptr_;
    hptr_ = NULL;
  }
}

template <class T>
void DeviceArray2D<T>::allocDevice()
{
  // width in bytes
  size_t width_bytes = width_ * sizeof(T);

  checkCudaErrors(cudaMallocPitch(reinterpret_cast<void **>(&dptr_[0]), &pitch_, width_bytes, height_));

  if(use_double_buffer_)
  {
    checkCudaErrors(cudaMallocPitch(reinterpret_cast<void **>(&dptr_[1]), &pitch_, width_bytes, height_));
  }
}

template <class T>
void DeviceArray2D<T>::freeDevice()
{
  if(dptr_[0] != NULL)
  {
    checkCudaErrors(cudaFree(dptr_[0]));
    dptr_[0] = NULL;
  }

  if(dptr_[1] != NULL)
  {
    checkCudaErrors(cudaFree(dptr_[1]));
    dptr_[1] = NULL;
  }
}

template <class T>
void DeviceArray2D<T>::swap()
{
  std::swap(current_read_, current_write_);
}

template <class T>
void DeviceArray2D<T>::copy(Direction dir)
{
  if(!use_host_)
  {
    allocHost();
    use_host_ = true;
  }

  // width in bytes
  size_t width_bytes = width_ * sizeof(T);

  switch(dir)
  {
    case HOST_TO_DEVICE:
      checkCudaErrors(cudaMemcpy2D(reinterpret_cast<void *>(dptr_[current_read_]), pitch_, reinterpret_cast<void *>(hptr_), width_bytes, width_bytes, height_, cudaMemcpyHostToDevice));
      break;

    case DEVICE_TO_HOST:
      checkCudaErrors(cudaMemcpy2D(reinterpret_cast<void *>(hptr_), width_bytes, reinterpret_cast<void *>(dptr_[current_read_]), pitch_, width_bytes, height_, cudaMemcpyDeviceToHost));
  }
}

template <class T>
void DeviceArray2D<T>::upload(const T *hptr)
{
  // width in bytes
  size_t width_bytes = width_ * sizeof(T);

  checkCudaErrors(cudaMemcpy2D(reinterpret_cast<void *>(dptr_[current_read_]), pitch_, reinterpret_cast<const void *>(hptr), width_bytes, width_bytes, height_, cudaMemcpyHostToDevice));
}
} // namespace

#endif /* __CUGL_COMMON_CONTAINERS_DEVICE_ARRAY_IMPL_HPP__ */
