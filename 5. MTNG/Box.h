/*#ifndef _BOX_H_
#define _BOX_H_

#include "Shape.h"

class Box : Shape{
public:
	Box(Point3D centerPos, MetricUnit lengthX, MetricUnit lengthY, MetricUnit lengthZ);
	Box();
	virtual ~Box();
	virtual bool isContainsPoint(const Point3D &point) const;
	virtual bool getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const;
private:
	Point3D centerPos_;
	MetricUnit lengthX_;
	MetricUnit lengthY_;
	MetricUnit lengthZ_;
};

#endif

*/