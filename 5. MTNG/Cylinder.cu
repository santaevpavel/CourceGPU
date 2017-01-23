#include "Cylinder.h"
#include <math.h>
#include <limits>

#define DOUBLE_EPSOLON 0.00000000001

//__device__ __host__ Point3D getNearestPoint(Point3D from, Point3D p1, Point3D p2);

__device__ __host__ Cylinder::Cylinder(const Point3D &centerPos, MetricUnit radiusMax,
	MetricUnit radiusMin, MetricUnit halfHeight)
	:centerPos_(centerPos), radiusMin_(radiusMin), 
	radiusMax_(radiusMax), halfHeight_(halfHeight) {
}

__device__ __host__ Cylinder::Cylinder(){
	Cylinder(Point3D(0, 0, 0), 1, 0.5, 1);
}

__device__ __host__ Cylinder::~Cylinder(){}

__device__ __host__ bool Cylinder::isContainsPoint(const Point3D &point) const {
	Line axis(centerPos_, Vector3D(0, 0, 1));
	// TODO optimize
	return axis.getMinDistPoint(point).distance2To(centerPos_) < halfHeight_ * halfHeight_
		&& axis.getMinDistPoint(point).distance2To(point) < radiusMax_ * radiusMax_;
		//&& axis.getMinDistPoint(point).distance2To(point) > radiusMin_ * radiusMin_;
}

__device__ __host__ bool Cylinder::isContainsPointXY(const Point3D &point) const {
	Line axis(centerPos_, Vector3D(0, 0, 1));
	return axis.getMinDistPoint(point).distance2To(point) < radiusMax_ * radiusMax_;
}

__device__ __host__ bool Cylinder::isContainsPointZ(const Point3D &point) const {
	Line axis(centerPos_, Vector3D(0, 0, 1));
	return axis.getMinDistPoint(point).distance2To(centerPos_) < halfHeight_ * halfHeight_;
}

__device__ __host__ bool Cylinder::getZIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const{
	//if (abs(direction.z) < std::numeric_limits<double>::epsilon()){
	if (abs(direction.z) < DOUBLE_EPSOLON){
		*resPoint = initPoint + direction * 9999999999;
		return true;
	}
	double t1 = (centerPos_.z + halfHeight_ - initPoint.z) / direction.z;
	double t2 = (centerPos_.z - halfHeight_ - initPoint.z) / direction.z;
	double t;
	if (isContainsPointZ(initPoint)){
		t = t1 > 0 ? t1 : t2;
	} else {
		t = t1 < t2 ? t1 : t2;
		if (t < 0){
			return false;
		}
	}
	*resPoint = initPoint + direction * t;
	//*resPoint = *resPoint + direction*0.000000001;
	return true;
}

__device__ __host__ bool Cylinder::getXYIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const{
	Point3D initPointZ(initPoint.x, initPoint.y, 0);
	Vector3D directionZ(direction.x, direction.y, 0);
	Line lineZ(initPointZ, directionZ);
	Point3D centerZ(centerPos_.x, centerPos_.y, 0);
	Point3D normalPointZ = lineZ.getMinDistPoint(centerZ);
	if (0 == direction.x && 0 == direction.y){
		*resPoint = initPoint + direction * 100000000;
		return true;
	}
	if (!(initPointZ.distance2To(centerZ) < radiusMax_ * radiusMax_) && directionZ.scalarMultiple(centerZ - initPointZ) < 0){
		return false;
	}
	if (normalPointZ.distanceTo(centerZ) < radiusMax_){
		MetricUnit k = sqrt((radiusMax_* radiusMax_ - normalPointZ.distance2To(centerZ)) / (directionZ.length() * directionZ.length()));
		k = initPointZ.distance2To(centerZ) < radiusMax_ * radiusMax_ ? -k : k;
		Point3D xyPoint = normalPointZ + (directionZ * -k);
		double t;
		if (abs(initPointZ.x - xyPoint.x) > 0){
			t = (xyPoint.x - initPointZ.x) / directionZ.x;
		} else {
			t = (xyPoint.y - initPointZ.y) / directionZ.y;
		}
		*resPoint = initPoint + direction * t;
		//*resPoint = *resPoint + direction*0.00000001;
		return true;
	}
	return false;
}

__device__ bool Cylinder::getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const{
	Point3D shape1IntersecPoint;
	Point3D shape2IntersecPoint;
	bool isIntersecShape1 = getXYIntersectionPoint(initPoint, direction, &shape1IntersecPoint);
	bool isIntersecShape2 = getZIntersectionPoint(initPoint, direction, &shape2IntersecPoint);
	if (isIntersecShape1 && isIntersecShape2){
		if (isContainsPointXY(shape2IntersecPoint)
			&& isContainsPointZ(shape1IntersecPoint)){
			*resPoint = initPoint.getNearestPoint(shape1IntersecPoint, shape2IntersecPoint);
			*resPoint = *resPoint + direction*0.000000001;
		}
		else if (isContainsPointXY(shape2IntersecPoint)){
			*resPoint = shape2IntersecPoint;
			*resPoint = *resPoint + direction*0.000000001;
		}
		else if (isContainsPointZ(shape1IntersecPoint)){
			*resPoint = shape1IntersecPoint;
			*resPoint = *resPoint + direction*0.000000001;
		}
		else {
			return false;
		}
		return true;
	}
	else {
		return false;
	}

}

/*__device__ __host__ std::string Cylinder::ToString(){
	std::ostringstream str;
	str << "Cylinder [center = [" << centerPos_.toString() << "]] [rad = " << radiusMax_ << "] " << "[halfHeight = " << halfHeight_ << "]";
	return str.str();
	return "";
}*/