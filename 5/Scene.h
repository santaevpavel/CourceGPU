#ifndef __MTNG_SCENE_H__
#define __MTNG_SCENE_H__

#include "Shape.h"
#include "Color.h"

class Scene{
public:
	__device__ __host__ Scene(Shape **shapes, int size, Point3D light) : 
	shapes(shapes), size(size), light(light){}
	__device__ __host__ ~Scene(){}

	Shape **shapes;
	int size;
	Point3D light;

	__device__ Color trace(Line line){
		Point3D outPoint;

		bool isIntersecChild = tracePoint(line, &outPoint);

		if (isIntersecChild){
			double dis = (light-outPoint).length();
			double k = 255.0 * calcLightPower(dis);
			Point3D outPoint2;
			bool isIntersecViaLight = tracePoint(Line(outPoint, light - outPoint), &outPoint2);			
			if (!isIntersecViaLight || outPoint.distance2To(light) < outPoint.distance2To(outPoint2)){
				return Color(0, k, 0);
			} else {
				return Color(0, 0, 0);	
			}
		} else {
			return Color(0, 0, 0);
		}
	}

	__device__ bool tracePoint(Line line, Point3D * resPoint){

		Point3D initPoint = line.getStartPoint();
		Vector3D direction = line.getDirection();

		Point3D outPoint;
		Point3D tmpPoint;
		MetricUnit min = 9999;
		//min = initPoint.distance2To(tmpPoint);
		outPoint = tmpPoint;

		bool isIntersecChild = false;
		for (int j = 0; j < size; j++){
			Shape * i = shapes[j];
			if (i->getIntersectionPoint(initPoint, direction, &tmpPoint)){
				MetricUnit dist = initPoint.distance2To(tmpPoint);
				if (dist < min){
					isIntersecChild = true;
					min = dist;
					outPoint = tmpPoint;
				}
			}
		}
		*resPoint = outPoint;
		if (isIntersecChild){
			return true;	
		} else {
			return false;
		}
	}
	__device__ double calcLightPower(double dis){
		double min = 1;
		//double max = 100;
		if (min > dis){
			return 1;
		}
		double k = 1 / (dis * dis);
		return k;
	}
};

#endif
