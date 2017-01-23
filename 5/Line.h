#ifndef _LINE_H_
#define _LINE_H_

#include "Point3D.h"
#include "Vector3D.h"

class Line{
public:
	__host__ __device__ Line(const Point3D &startPoint, const Vector3D &direction)
		:startPoint_(startPoint), direction_(direction){
		direction_.normalize();
	}
	__host__ __device__ ~Line(){}
	__device__ __host__ inline Point3D getStartPoint() const{
		return startPoint_;
	}
	__device__ __host__ inline Vector3D getDirection() const{
		return direction_;
	}
	__device__ __host__ inline Point3D getMinDistPoint(const Point3D &point) const{
		/*MetricUnit k = ((point.x - startPoint_.x) * direction_.x + (point.y - startPoint_.y) * direction_.y + (point.z - startPoint_.z) * direction_.z)
			/ (direction_.x * direction_.x + direction_.y * direction_.y + direction_.z * direction_.z);
		return startPoint_.add(direction_.multiple(k));*/
		return startPoint_ + direction_*direction_.scalarMultiple(point - startPoint_);
	}
	/*Point3D getIntercesPoint(Line other) const{
		return Point3D(0, 0, 0);
	}*/
private:
	Point3D startPoint_;
	Vector3D direction_;
};

#endif