#ifndef _POINT_H_
#define _POINT_H_

#include "Util.h"
#include "Vector3D.h"
#include <math.h>
#include <sstream>


class Point3D{
public:
	MetricUnit x;
	MetricUnit y;
	MetricUnit z;
public:
	__device__ __host__ Point3D() :x(0), y(0), z(0){}
	__device__ __host__ Point3D(MetricUnit nx, MetricUnit ny, MetricUnit nz) 
		:x(nx), y(ny), z(nz){}
	__device__ __host__ ~Point3D(){};
public:
	__device__ __host__ MetricUnit distanceTo(const Point3D & point) const{
		return sqrt((point.x - x) * (point.x - x) + (point.y - y) * (point.y - y) + (point.z - z) * (point.z - z));
	}
	// Расстояние в квадрате
	__device__ __host__ MetricUnit distance2To(const Point3D & point) const{
		return(point.x - x) * (point.x - x) + (point.y - y) * (point.y - y) + (point.z - z) * (point.z - z);
	}
	__device__ __host__ Point3D multiple(MetricUnit k) const{
		return Point3D(x * k, y * k, z * k);
	}
	__device__ __host__ Point3D add(const Point3D & other) const{
		return Point3D(x + other.x, y + other.y, z + other.z);
	}
	__device__ __host__ Point3D substruct(const Point3D & other) const{
		return Point3D(x - other.x, y - other.y, z - other.z);
	}
	__device__ __host__ MetricUnit length() const{
		return distanceTo(Point3D(0, 0, 0));
	}
	__device__ __host__ Point3D operator+(const Vector3D &vector) const{
		return Point3D(x + vector.x, y + vector.y, z + vector.z);
	}
	__device__ __host__ Vector3D operator-(const Point3D & other) const{
		return Vector3D(x - other.x, y - other.y, z - other.z);
	}

	__device__ __host__ Point3D getNearestPoint(Point3D p1, Point3D p2) const{
		return distance2To(p1) < distance2To(p2) ? p1 : p2;
	}

	__device__ __host__ bool operator ==(const Point3D & other){
		return distance2To(other) < GEOM_EPS;
	}

	__device__ __host__ std::string toString(){
		/*std::ostringstream str;
		str << "x = " << (double)x << " y = " << (double)y << " z = " << (double)z;
		return str.str();*/
		return "";
	}

};
#endif