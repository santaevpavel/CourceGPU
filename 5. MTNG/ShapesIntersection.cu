#include "ShapesIntersection.h"

//Point3D getNearestPoint(Point3D from, Point3D p1, Point3D p2);

__device__ __host__ ShapesIntersection::ShapesIntersection(Shape * shape1, Shape * shape2){
	shape1_ = shape1;
	shape2_ = shape2;
}

__device__ __host__ ShapesIntersection::~ShapesIntersection(){}

__device__ bool ShapesIntersection::isContainsPoint(const Point3D &point) const{
	return shape1_->isContainsPoint(point) && shape2_->isContainsPoint(point);
}

__device__ bool ShapesIntersection::getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const{
	Point3D shape1IntersecPoint;
	Point3D shape2IntersecPoint;
	bool isIntersecShape1 = shape1_->getIntersectionPointInShape(initPoint, direction, &shape1IntersecPoint);
	bool isIntersecShape2 = shape2_->getIntersectionPointInShape(initPoint, direction, &shape2IntersecPoint);
	if (isIntersecShape1 && isIntersecShape2){
		if (shape1_->isContainsPoint(shape2IntersecPoint)
			&& shape2_->isContainsPoint(shape1IntersecPoint)){
			*resPoint = initPoint.getNearestPoint(shape1IntersecPoint, shape2IntersecPoint);
			*resPoint = *resPoint + direction*0.000000001;
		}
		else if (shape1_->isContainsPoint(shape2IntersecPoint)){
			*resPoint = shape2IntersecPoint;
			*resPoint = *resPoint + direction*0.000000001;
		}
		else if (shape2_->isContainsPoint(shape1IntersecPoint)){
			*resPoint = shape1IntersecPoint;
			*resPoint = *resPoint + direction*0.000000001;
		}
		else {
			if (isContainsPoint(initPoint)){
				*resPoint = shape1IntersecPoint;
				*resPoint = *resPoint + direction*0.000000001;
			}
			return false;
		}
		return true;
	}  else {
		return false;
	}
}

/*__device__ Point3D getNearestPoint2(Point3D from, Point3D p1, Point3D p2){
	if (p1.distanceTo(p2) < 0.0000001){

	}
	return from.distance2To(p1) < from.distance2To(p2) ? p1 : p2;
}*/