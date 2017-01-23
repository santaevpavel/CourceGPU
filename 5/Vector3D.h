#ifndef __MTNG_VECTOR_3D_H__
#define __MTNG_VECTOR_3D_H__

#include "Util.h"

class Vector3D{
public:
	MetricUnit x;
	MetricUnit y;
	MetricUnit z;
public:

	__host__ __device__ Vector3D(){}

	__host__ __device__ Vector3D(MetricUnit nx, MetricUnit ny, MetricUnit nz)
		:x(nx), y(ny), z(nz){
	}
	__host__ __device__ ~Vector3D(){}
public:
	__host__ __device__ void normalize(){
		MetricUnit l = length();
		x = x / l;
		y = y / l;
		z = z / l;
	}
	__host__ __device__ MetricUnit length() const{
		return sqrt(x * x + y * y + z * z);
	}

	__device__ __host__ MetricUnit scalarMultiple(const Vector3D & other) const{
		return other.x * x + other.y * y + other.z * z;
	}

	__device__ __host__ Vector3D operator*(MetricUnit val) const{
		return Vector3D(x * val, y * val, z * val);
	}

	__device__ __host__ Vector3D add(Vector3D other) const{
		return Vector3D(x + other.x, y + other.y, z + other.z);
	}

};
#endif