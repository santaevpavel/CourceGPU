#ifndef __MTNG_SHAPES_EXCLUSION_H_
#define __MTNG_SHAPES_EXCLUSION_H_

#include "Shape.h"

class ShapesExclusion : public Shape{
public:
	__device__ __host__ ShapesExclusion(Shape * shape1, Shape * shape2);
	__device__ __host__ virtual ~ShapesExclusion();
	__device__ __host__ virtual bool isContainsPoint(const Point3D &point) const;
	__device__ virtual bool getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const;
private:
	Shape * shape1_;
	Shape * shape2_;
};

#endif