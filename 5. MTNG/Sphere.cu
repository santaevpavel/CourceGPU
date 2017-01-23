#include "Sphere.h"
#include <math.h>

/*__device__ __host__ Sphere::Sphere(const Point3D centerPos, MetricUnit radius)
	:centerPos_(centerPos), radius_(radius){
}

__device__ __host__ Sphere::Sphere(){
	Sphere(Point3D(0, 0, 0), 1);
}

//__device__ __host__ Sphere::~Sphere(){}

__device__ __host__ bool Sphere::isContainsPoint(const Point3D &point) const {
	return radius_*radius_ > point.distance2To(centerPos_);
}

__device__ bool Sphere::getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const{
	Line line(initPoint, direction);
	Point3D normalPoint = line.getMinDistPoint(centerPos_);
	//Отнормированное направление на 1
	const Vector3D dir=line.getDirection();
	const Vector3D direction_to_center = centerPos_ - initPoint;

	//Проверяем что луч направлен в сторону сферы
	if (!Sphere::isContainsPoint(initPoint) && dir.scalarMultiple(direction_to_center) < 0){
		return false;
	}

	if (radius_ > normalPoint.distanceTo(centerPos_)){
		MetricUnit k = sqrt((radius_* radius_ - normalPoint.distance2To(centerPos_)));
		k = isContainsPoint(initPoint) ? k : -k;
		Point3D res = normalPoint + (dir * k);
		resPoint->x = res.x;
		resPoint->y = res.y;
		resPoint->z = res.z;
		return true;
	}
	return false;
}*/
