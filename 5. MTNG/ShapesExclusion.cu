#include "ShapesExclusion.h"

__device__ Point3D getFurtherPoint(Point3D from, Point3D p1, Point3D p2);

__device__ __host__ ShapesExclusion::ShapesExclusion(Shape * shape1, Shape * shape2){
	shape1_ = shape1;
	shape2_ = shape2;
}

__device__ __host__ ShapesExclusion::~ShapesExclusion(){}

__device__ bool ShapesExclusion::isContainsPoint(const Point3D &point) const{
	return shape1_->isContainsPoint(point) && !shape2_->isContainsPoint(point);
}

__device__ bool ShapesExclusion::getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const{
	Point3D shape1IntersecPoint;
	Point3D shape2IntersecPoint;
	bool isIntersecShape1 = shape1_->getIntersectionPointInShape(initPoint, direction, &shape1IntersecPoint);
	bool isIntersecShape2 = shape2_->getIntersectionPointInShape(initPoint, direction, &shape2IntersecPoint);
	if (isIntersecShape1 && isIntersecShape2){
		if (!isContainsPoint(shape2IntersecPoint)
			&& !isContainsPoint(shape1IntersecPoint)){

			if (isContainsPoint(initPoint)){
				*resPoint = initPoint.getNearestPoint(shape1IntersecPoint, shape2IntersecPoint);
			} else {
				*resPoint = getFurtherPoint(initPoint, shape1IntersecPoint, shape2IntersecPoint);
			}
			
			//*resPoint = *resPoint + direction*0.000000001;
		} else if (isContainsPoint(shape1IntersecPoint)){
			*resPoint = shape1IntersecPoint;
			//*resPoint = *resPoint + direction*0.000000001;
		} else if (isContainsPoint(shape2IntersecPoint)){
			*resPoint = shape2IntersecPoint;
			//*resPoint = *resPoint + direction*0.000000001;
		} else {
			return false;
		}
		return true;
	} else if (isIntersecShape1){
		*resPoint = shape1IntersecPoint;
		return true;
	} else if (isIntersecShape2){
		return false;
	} else {
		return false;
	}
}

__device__ Point3D getFurtherPoint(Point3D from, Point3D p1, Point3D p2){
	return from.distance2To(p1) > from.distance2To(p2) ? p1 : p2;
}