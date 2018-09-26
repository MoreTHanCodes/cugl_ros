#ifndef __CUGL_COMMON_CONTAINERS_H__
#define __CUGL_COMMON_CONTAINERS_H__

#include <cugl_ros/common/containers/device_array.h>
#include <cugl_ros/common/containers/texture_bridge.h>

namespace cugl
{
typedef DeviceArray<float4>           VertexArray;
typedef DeviceArray<float4>           NormalArray;
typedef DeviceArray<float4>           ColorArray;
typedef DeviceArray2D<float4>         VertexImage;
typedef DeviceArray2D<float4>         NormalImage;
typedef DeviceArray2D<unsigned char>  ColorImage8u;
typedef DeviceArray2D<uchar4>         ColorImage8u4c;
typedef DeviceArray2D<unsigned short> DepthImage16u;
typedef DeviceArray2D<float>          DepthImage32f;
} // namespace

#endif /* __CUGL_COMMON_CONTAINERS_H__ */
