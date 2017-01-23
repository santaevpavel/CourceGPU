#ifndef __MTNG_SHAPES_INTERSECTION_H_
#define __MTNG_SHAPES_INTERSECTION_H_

#include "Shape.h"

class ShapesIntersection : public Shape{
public:
	__device__ __host__ ShapesIntersection(Shape * shape1, Shape * shape2);
	__device__ __host__ virtual ~ShapesIntersection();
	__device__ __host__ virtual bool isContainsPoint(const Point3D &point) const;
	__device__ virtual bool getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const;
private:
	Shape * shape1_;
	Shape * shape2_;
};

#endif