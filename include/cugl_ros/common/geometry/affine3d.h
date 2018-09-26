#ifndef __CUGL_COMMON_GEOMETRY_AFFINE3D_H__
#define __CUGL_COMMON_GEOMETRY_AFFINE3D_H__

#if defined(__CUDACC__)
  #define __CUGL_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
  #define __CUGL_HOST_DEVICE__
#endif

#include <cugl_ros/common/helper_math.h>

namespace cugl
{
/*
 * Class to represent a 3d affine transformation.
 * The elements are stored in colume major order.
 */
class Affine3d
{
 public:
  __CUGL_HOST_DEVICE__ Affine3d() { makeIdentity(); }
  
  __CUGL_HOST_DEVICE__ Affine3d(float s) { setValue(s); }
  
  __CUGL_HOST_DEVICE__ Affine3d(const float *mp) { setValue(mp); }

  __CUGL_HOST_DEVICE__ Affine3d(const float4 *vp) { setValue(vp); }
  
  __CUGL_HOST_DEVICE__ Affine3d(const Affine3d &other)
  {
    for (int j = 0; j < 4; j++)
    {
      #pragma unroll
      for (int i = 0; i < 4; i++)
      {
        this->at(i, j) = other.at(i, j);
      }
    }
  }

  __CUGL_HOST_DEVICE__ Affine3d &operator = (const Affine3d &other)
  {
    if (this != &other)
    {
      for (int j = 0; j < 4; j++)
      {
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
          this->at(i, j) = other.at(i, j);
        }
      }
    }
  
    return *this;
  }

  __CUGL_HOST_DEVICE__ void getValue(float *mp) const
  {
    int c = 0;
  
    for (int j = 0; j < 4; j++)
    {
      #pragma unroll
      for (int i = 0; i < 4; i++)
      {
        mp[c++] = this->at(i, j);
      }
    }
  }

  __CUGL_HOST_DEVICE__ void getValue(float4 *vp) const
  {
    #pragma unroll
    for (int c = 0; c < 4; c++)
    {
      vp[c] = make_float4(this->at(0, c), this->at(1, c), this->at(2, c), this->at(3, c));
    }
  }

  __CUGL_HOST_DEVICE__ void getTranslate(float3 &t) const
  {
    t.x = this->at(0, 3);
    t.y = this->at(1, 3);
    t.z = this->at(2, 3);
  }

  __CUGL_HOST_DEVICE__ void getRotate(float3 r[3]) const
  {
    #pragma unroll
    for (int col = 0; col < 3; col++)
    {
      r[col].x = this->at(0, col);
      r[col].y = this->at(1, col);
      r[col].z = this->at(2, col);
    }
  }

  __CUGL_HOST_DEVICE__ void setValue(float s)
  {
    for(int j = 0; j < 4; j++)
    {
      #pragma unroll
      for(int i = 0; i < 4; i++)
      {
        this->at(i, j) = s;
      }
    }
  }

  __CUGL_HOST_DEVICE__ void setValue(const float *mp)
  {
    int c = 0;
  
    for (int j = 0; j < 4; j++)
    {
      #pragma unroll
      for (int i = 0; i < 4; i++)
      {
        this->at(i, j) = mp[c++];
      }
    }
  }

  __CUGL_HOST_DEVICE__ void setValue(const float4 *vp)
  {
    #pragma unroll
    for (int c = 0; c < 4; c++)
    {
      this->setColumn(c, vp[c]);
    }
  }

  __CUGL_HOST_DEVICE__ void makeIdentity()
  {
    data[0] = 1.0f;
    data[1] = 0.0f;
    data[2] = 0.0f;
    data[3] = 0.0f;
  
    data[4] = 0.0f;
    data[5] = 1.0f;
    data[6] = 0.0f;
    data[7] = 0.0f;
  
    data[8] = 0.0f;
    data[9] = 0.0f;
    data[10] = 1.0f;
    data[11] = 0.0f;
  
    data[12] = 0.0f;
    data[13] = 0.0f;
    data[14] = 0.0f;
    data[15] = 1.0f;
  }

  __CUGL_HOST_DEVICE__ void setScale(float s)
  {
    #pragma unroll
    for(int i = 0; i < 3; i++)
    {
      this->at(i, i) = s;
    }
  }

  __CUGL_HOST_DEVICE__ void setScale(const float *sp)
  {
    #pragma unroll
    for(int i = 0; i < 3; i++)
    {
      this->at(i, i) = sp[i];
    }
  }

  // __CUGL_HOST_DEVICE__ void setTranslate(const float *tp)
  // {
  //   #pragma unroll
  //   for(int i = 0; i < 3; i++)
  //   {
  //     this->at(i, 3) = tp[i];
  //   }
  // }

  // __CUGL_HOST_DEVICE__ void setRotate(const float *rp)
  // {
  //   int c = 0;
  // 
  //   for(int j = 0; j < 3; j++)
  //   {
  //     #pragma unroll
  //     for(int i = 0; i < 3; i++)
  //     {
  //       this->at(i, j) = rp[c++];
  //     }
  //   }
  // }

  __CUGL_HOST_DEVICE__ void setRow(int r, const float4 &v)
  {
    this->at(r, 0) = v.x;
    this->at(r, 1) = v.y;
    this->at(r, 2) = v.z;
    this->at(r, 3) = v.w;
  }

  __CUGL_HOST_DEVICE__ void setColumn(int c, const float4 &v)
  {
    this->at(0, c) = v.x;
    this->at(1, c) = v.y;
    this->at(2, c) = v.z;
    this->at(3, c) = v.w;
  }

  __CUGL_HOST_DEVICE__ float &at(int row, int col)
  {
    return data[row | (col<<2)];
  }

  __CUGL_HOST_DEVICE__ const float &at(int row, int col) const
  {
    return data[row | (col<<2)];
  }

  ////////////////////////////////////////////////////////////
  // Math operations
  ////////////////////////////////////////////////////////////
  __CUGL_HOST_DEVICE__ Affine3d &operator *= (const Affine3d &rhs)
  {
    Affine3d lhs(*this);
    this->setValue(0.f);
  
    for(int i = 0; i < 4; i++)
    {
      for(int j = 0; j < 4; j++)
      {
        #pragma unroll
        for(int c = 0; c < 4; c++)
        {
          this->at(i, j) += lhs.at(i, c) * rhs.at(c, j);
        }
      }
    }
  
    return *this;
  }
  
  __CUGL_HOST_DEVICE__ friend Affine3d operator * (const Affine3d &lhs, const Affine3d &rhs)
  {
    Affine3d m(0.f);
  
    for(int i = 0; i < 4; i++)
    {
      for(int j = 0; j < 4; j++)
      {
        #pragma unroll
        for(int c = 0; c < 4; c++)
        {
          m.at(i, j) += lhs.at(i, c) * rhs.at(c, j);
        }
      }
    }
  
    return m;
  }
  
  // dst = M * src
  __CUGL_HOST_DEVICE__ friend float4 operator * (const Affine3d &m, const float4 &src)
  {
    float4 dst = make_float4(0.f);

    dst.x = (m.at(0, 0) * src.x + m.at(0, 1) * src.y +
             m.at(0, 2) * src.z + m.at(0, 3) * src.w);

    dst.y = (m.at(1, 0) * src.x + m.at(1, 1) * src.y +
             m.at(1, 2) * src.z + m.at(1, 3) * src.w);

    dst.z = (m.at(2, 0) * src.x + m.at(2, 1) * src.y +
             m.at(2, 2) * src.z + m.at(2, 3) * src.w);

    dst.w = (m.at(3, 0) * src.x + m.at(3, 1) * src.y +
             m.at(3, 2) * src.z + m.at(3, 3) * src.w);
  
    return dst;
  }
  
  // dst = src * M
  __CUGL_HOST_DEVICE__ friend float4 operator * (const float4 &src, const Affine3d &m)
  {
    float4 dst = make_float4(0.f);

    dst.x = (src.x * m.at(0, 0) + src.y * m.at(1, 0) +
             src.z * m.at(2, 0) + src.w * m.at(3, 0));

    dst.y = (src.x * m.at(0, 1) + src.y * m.at(1, 1) +
             src.z * m.at(2, 1) + src.w * m.at(3, 1));
  
    dst.z = (src.x * m.at(0, 2) + src.y * m.at(1, 2) +
             src.z * m.at(2, 2) + src.w * m.at(3, 2));
  
    dst.w = (src.x * m.at(0, 3) + src.y * m.at(1, 3) +
             src.z * m.at(2, 3) + src.w * m.at(3, 3));
  
    return dst;
  }
  
  __CUGL_HOST_DEVICE__ Affine3d &operator *= (const float &s)
  {
    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
      this->at(0, i) *= s;
      this->at(1, i) *= s;
      this->at(2, i) *= s;
      this->at(3, i) *= s;
    }
  
    return *this;
  }
  
  __CUGL_HOST_DEVICE__ friend Affine3d operator * (const Affine3d &lhs, const float &rhs)
  {
    Affine3d m(lhs);
    m *= rhs;
  
    return m;
  }
  
  __CUGL_HOST_DEVICE__ Affine3d &operator += (const Affine3d &rhs)
  {
    #pragma unroll
    for(int i = 0; i < 4; ++i)
    {
      this->at(0, i) += rhs.at(0, i);
      this->at(1, i) += rhs.at(1, i);
      this->at(2, i) += rhs.at(2, i);
      this->at(3, i) += rhs.at(3, i);
    }
  
    return *this;
  }
  
  __CUGL_HOST_DEVICE__ friend Affine3d operator + (const Affine3d &lhs, const Affine3d &rhs)
  {
    Affine3d m(lhs);
    m += rhs;
  
    return m;
  }

  __CUGL_HOST_DEVICE__ Affine3d &operator -= (const Affine3d &rhs)
  {
    #pragma unroll
    for(int i = 0; i < 4; ++i)
    {
      this->at(0, i) -= rhs.at(0, i);
      this->at(1, i) -= rhs.at(1, i);
      this->at(2, i) -= rhs.at(2, i);
      this->at(3, i) -= rhs.at(3, i);
    }
  
    return *this;
  }
  
  __CUGL_HOST_DEVICE__ friend Affine3d operator - (const Affine3d &lhs, const Affine3d &rhs)
  {
    Affine3d m(lhs);
    m -= rhs;
  
    return m;
  }

 private:
  float data[16];
};
}

#endif /* __CUGL_COMMON_GEOMETRY_AFFINE3D_H__ */
