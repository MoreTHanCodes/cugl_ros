#ifndef __CUGL_COMMON_CONTAINERS_DEVICE_ARRAY_HANDLE_H__
#define __CUGL_COMMON_CONTAINERS_DEVICE_ARRAY_HANDLE_H__

#if defined(__CUDACC__)
  #define __CUGL_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
  #define __CUGL_HOST_DEVICE__
#endif

#include <cstddef>

namespace cugl
{
/*
 * A handle class of a linear array in CPU/GPU.
 */
template <typename T>
struct DeviceArrayHandle
{
  __CUGL_HOST_DEVICE__ DeviceArrayHandle() : data(0) {}
  __CUGL_HOST_DEVICE__ DeviceArrayHandle(T *data_arg) : data(data_arg) {}

  __CUGL_HOST_DEVICE__ ~DeviceArrayHandle() { data = 0; }

  __CUGL_HOST_DEVICE__ DeviceArrayHandle &operator = (DeviceArrayHandle &other)
  {
    data = other.ptr();

    return *this;
  }

  __CUGL_HOST_DEVICE__       T *ptr()       { return data; }
  __CUGL_HOST_DEVICE__ const T *ptr() const { return reinterpret_cast<const T*>(data); }

  __CUGL_HOST_DEVICE__       T &at(int x)       { return ptr()[x]; }
  __CUGL_HOST_DEVICE__ const T &at(int x) const { return ptr()[x]; }

  T *data;
};

/*
 * A handle class of a 2D linear array in CPU/GPU.
 */
template <typename T>
struct DeviceArrayHandle2D
{
  __CUGL_HOST_DEVICE__ DeviceArrayHandle2D() : data(0), stride(0) {}
  __CUGL_HOST_DEVICE__ DeviceArrayHandle2D(T *data_arg, size_t stride_arg) : data(data_arg), stride(stride_arg) {}

  __CUGL_HOST_DEVICE__ ~DeviceArrayHandle2D() { data = 0; stride = 0; }

  __CUGL_HOST_DEVICE__ DeviceArrayHandle2D &operator = (DeviceArrayHandle2D &other)
  {
    data = other.ptr();
    stride = other.stride;

    return *this;
  }

  __CUGL_HOST_DEVICE__       T *ptr(int y = 0)       { return reinterpret_cast<T *>(reinterpret_cast<char *>(data) + y * stride); }
  __CUGL_HOST_DEVICE__ const T *ptr(int y = 0) const { return reinterpret_cast<const T *>(reinterpret_cast<const char *>(data) + y * stride); }

  __CUGL_HOST_DEVICE__       T &at(int y, int x)       { return ptr(y)[x]; }
  __CUGL_HOST_DEVICE__ const T &at(int y, int x) const { return ptr(y)[x]; }

  T *data;
  size_t stride; // stride between two consecutive rows in bytes
};
} // namespace

#endif /* __CUGL_COMMON_CONTAINERS_DEVICE_ARRAY_HANDLE_H__ */
