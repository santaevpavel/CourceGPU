#ifndef __MTNG_SPHERE_H__
#define __MTNG_SPHERE_H__

#include "Shape.h"
#include <sstream>

class Sphere: public Shape{
public:
	//__device__ __host__ Sphere(const Point3D centerPos, MetricUnit radius);
	//__device__ __host__ Sphere();
	__device__ __host__ Sphere(const Point3D centerPos, MetricUnit radius)
		:centerPos_(centerPos), radius_(radius){
	}

	/*__device__ __host__ Sphere(){
		Sphere(Point3D(0, 0, 0), 1);
	}*/

	__device__ __host__ virtual ~Sphere(){};

	//__device__ __host__ virtual bool isContainsPoint(const Point3D &point) const;
	//__device__ virtual bool getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const;
	//virtual bool getIntersectionPoint(Line line, Point3D & outPoint);

	/*__device__ __host__ virtual std::string ToString(){
		std::ostringstream str;
		str << "Sphere [center = [" << centerPos_.toString() << "]] [rad = " << radius_ << "]";
		return str.str();	
		return "";
	}*/

	__device__ __host__ bool isContainsPoint(const Point3D &point) const {
		return radius_*radius_ > point.distance2To(centerPos_);
	}

	__device__ bool getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const{
		Line line(initPoint, direction);
		Point3D normalPoint = line.getMinDistPoint(centerPos_);
		//Îòíîðìèðîâàííîå íàïðàâëåíèå íà 1
		const Vector3D dir=line.getDirection();
		const Vector3D direction_to_center = centerPos_ - initPoint;

		//Ïðîâåðÿåì ÷òî ëó÷ íàïðàâëåí â ñòîðîíó ñôåðû
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
	}

private:
	Point3D centerPos_;
	MetricUnit radius_;
};

#endif