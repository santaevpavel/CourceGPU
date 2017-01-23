#ifndef __MTNG_SCENE_H__
#define __MTNG_SCENE_H__

#include "Shape.h"

class Scene{
public:
	__device__ __host__ Scene(Shape * rootShape): root(rootShape){}
	__device__ __host__ ~Scene(){
		//delete root;
	}
	__device__ __host__ Shape * getContainsShape(const Point3D & point) const{
		return root->getContainShape(point);
	}
	Shape * root;
private:
	
};

#endif
