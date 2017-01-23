#ifndef _SHAPES_MERGE_H_
#define _SHAPES_MERGE_H_

#include "Shape.h"

class ShapesUnion : public Shape{
public:
	__device__ __host__ ShapesUnion(Shape * shape1, Shape * shape2);
	__device__ __host__ virtual ~ShapesUnion();
	__device__ __host__ virtual bool isContainsPoint(const Point3D &point) const;
	__device__ virtual bool getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const;
private:
	Shape * shape1_;
	Shape * shape2_;
};

#endif