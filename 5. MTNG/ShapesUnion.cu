#include "ShapesUnion.h"

//Point3D getNearestPoint(Point3D from, Point3D p1, Point3D p2);

__device__ __host__ ShapesUnion::ShapesUnion(Shape * shape1, Shape * shape2){
	shape1_ = shape1;
	shape2_ = shape2;
}

__device__ __host__ ShapesUnion::~ShapesUnion(){}

__device__ bool ShapesUnion::isContainsPoint(const Point3D &point) const{
	return shape1_->isContainsPoint(point) || shape2_->isContainsPoint(point);
}

__device__ bool ShapesUnion::getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const{
	Point3D shape1IntersecPoint;
	Point3D shape2IntersecPoint;
	bool isIntersecShape1 = shape1_->getIntersectionPointInShape(initPoint, direction, &shape1IntersecPoint);
	bool isIntersecShape2 = shape2_->getIntersectionPointInShape(initPoint, direction, &shape2IntersecPoint);
	if (!isIntersecShape1 && !isIntersecShape2){
		return false;
	} else if (isIntersecShape1 && !isIntersecShape2){
		*resPoint = shape1IntersecPoint;
		*resPoint = *resPoint + direction*0.000000001;
	} else if (!isIntersecShape1 && isIntersecShape2){
		*resPoint = shape2IntersecPoint;
		*resPoint = *resPoint + direction*0.000000001;
	} else {
		*resPoint = initPoint.getNearestPoint(shape1IntersecPoint, shape2IntersecPoint);
		*resPoint = *resPoint + direction*0.000000001;
	}
	return true;
}
