#ifndef __MTNG_CYLINDER_H__
#define __MTNG_CYLINDER_H__

#include "Shape.h"

class Cylinder : public Shape{
public:
	__device__ __host__ Cylinder(const Point3D &centerPos, MetricUnit radiusMax,
		MetricUnit radiusMin, MetricUnit halfHeight);
	__device__ __host__ Cylinder();
	__device__ __host__ virtual ~Cylinder();
	__device__ __host__ virtual bool isContainsPoint(const Point3D &point) const;
	__device__ virtual bool getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const;
	//__device__ __host__ virtual std::string ToString();
private:
	__device__ __host__ bool getXYIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint1) const;
	__device__ __host__ bool getZIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint1) const;
	__device__ __host__ bool isContainsPointXY(const Point3D &point) const;
	__device__ __host__ bool isContainsPointZ(const Point3D &point) const;
private:
	Point3D centerPos_;
	MetricUnit radiusMin_;
	MetricUnit radiusMax_;
	MetricUnit halfHeight_;
};

#endif