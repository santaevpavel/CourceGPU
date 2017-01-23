#ifndef _SHAPE_H_
#define _SHAPE_H_

#include "Point3D.h"
#include "Vector3D.h"
#include "Line.h"

class Shape{

public:
	__device__ __host__ Shape(){
	}
	__device__ __host__ virtual ~Shape(){}
	__device__ virtual bool getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const = 0;

};
#endif