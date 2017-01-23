/*#include "Box.h"

Box::Box(Point3D centerPos, MetricUnit lengthX, MetricUnit lengthY, MetricUnit lengthZ){
	centerPos_ = centerPos;
	lengthX_ = lengthX;
	lengthY_ = lengthY;
	lengthZ_ = lengthZ;
}

Box::Box(){
	Box(Point3D(0, 0, 0), 1, 1, 1);
}

Box::~Box(){}

bool Box::isContainsPoint(const Point3D &point) const{
	return centerPos_.x + lengthX_ / 2 < point.x && centerPos_.x - lengthX_ / 2 > point.x
		&& centerPos_.y + lengthY_ / 2 < point.y && centerPos_.y - lengthY_ / 2 > point.y
		&& centerPos_.z + lengthZ_ / 2 < point.z && centerPos_.z - lengthZ_ / 2 > point.z;
}

bool Box::getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const{
	
	return false;
}*/