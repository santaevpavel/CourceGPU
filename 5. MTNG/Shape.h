#ifndef _SHAPE_H_
#define _SHAPE_H_

#include "Point3D.h"
#include "Vector3D.h"
#include "Line.h"
#include "Material.h"
#include <list>
#include <exception>

class Shape{

public:
	__device__ __host__ Shape(){
		parent = nullptr;
		childs = new Shape*[10];
		size = 0;
	}
	__device__ __host__ virtual ~Shape(){
		// Êàê-sî 
		// óäàëÿsü âñåõ äåsåé
	}
protected:
	// Âîçðàùàås true åñëè åñsü ïåðåñå÷åíèå.
	// Â resPoint âîçâðàùàås sî÷êó ïåðåñå÷åíèÿ
	__device__ virtual bool getIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint) const = 0;

public:
	//! Ñîäåðæèsñÿ sî÷êà âíósðè îáúåêsà 
	__device__ __host__ virtual bool isContainsPoint(const Point3D &point) const = 0;

	// Íàõîäès îáúåês â êîsîðîì íàõîäèsñÿ sî÷êà (èùås â äî÷åðíèõ)
	// Âîçâðàùàås nullptr åñëè sî÷êà íå íàõîäèsñÿ â ýsîì îáúåêså, 
	// âîçðàùàås ñåáÿ åñëè sî÷êà íå íàõîäèsñÿ â äî÷åíèõ ýëåìåísàõ,
	// â äðóãîì ñëó÷àå âîçðàùàås äî÷åðíèé îáúåês â êîsîðîì íàõîäèsüñÿ ýsîs îáúåês
	__device__ __host__ Shape * getContainShape(const Point3D &point) const {
		//return NULL;
		if (isContainsPoint(point)){
			for (int j = 0; j < size; j++){
			Shape * i = childs[j];
			//for (std::list<Shape * >::const_iterator i = childs.begin(); i != childs.end(); i++){
				Shape * tmp = i->getContainShape(point);
				if (nullptr != tmp) {
					return tmp;
				}
			}
			return (Shape *)this;
		} else {
			return nullptr;
		}
	}

	// Íàõîäès îáúåês â êîsîðîì íàõîäèsñÿ sî÷êà (èùås â ðîäèsåëüñêèõ)
	// Âîçâðàùàås ñåáÿ åñëè sî÷êà íàõîäèsñÿ â ýsîì îáúåêså, 
	// , âîçðàùàås nullptr åñëè sî÷êà íå íàõîäèsñÿ ðîäèsåëüêèõ ýëåìåísàõ,
	// â äðóãîì âîçðàùàås ðîäèsåëüêèé îáúåês â êîsîðîì íàõîäèsüñÿ ýsîs îáúåês
	__device__ __host__ Shape * getContainParentShape(const Point3D &point) const {
		if (!isContainsPoint(point)){
			if (nullptr == parent){
				return nullptr;
			}
			return parent->getContainParentShape(point);
		} else {
			return (Shape *)this;
		}
	}

	// Âîçðàùàås true åñëè åñsü ïåðåñå÷åíèå.
	// Â resPoint âîçâðàùàås sî÷êó ïåðåñå÷åíèÿ, êîsîðàÿ íàõîäèsñÿ â âíósðè îáúåêsêà
	__device__ bool getIntersectionPointInShape(const Point3D & initPoint, const Vector3D &direction, Point3D* resPoint){
		Point3D tmpPoint;
		//bool isIntersec;
		bool isInside = isContainsPoint(initPoint);
		if (getIntersectionPoint(initPoint, direction, &tmpPoint)){
			Point3D offsetPoint = tmpPoint + (direction * SMALL_DIST);
			if (isInside != isContainsPoint(offsetPoint)){
				*resPoint = offsetPoint;
				return true;
			} else {
				// åñëè ïðè ñìåùåíèå âûøëè çà ïðåäåëû îáúåêsà
				return getIntersectionPointInShape(offsetPoint, direction, resPoint);
			}
		}
		return false;
	}
	// Íàõîäès áëèæàéøåå ïåðåñå÷åíèå. 
	// InitPoint äîëæåí íàõîäèsñÿ âíósðè îáúåêsà
	// Âîçâðàùàås sî÷êó è îáúåês, â êîsîðîì íàõîäèsñÿ ýsà sî÷êà
	__device__ void getNearIntersectionPoint(const Point3D & initPoint, const Vector3D &direction, MetricUnit* passedDist, Point3D* resPoint, Shape **resShape){
		Shape *outShape;
		Point3D outPoint;
		Point3D tmpPoint;
		MetricUnit min = 0;
		if (!getIntersectionPointInShape(initPoint, direction, &tmpPoint)){
			//std::cout << "Error in getNearIntersectionPoint:90" << std::cout;
			//throw std::string("Îøèáêà: initPoint äîëæåí ëåæàsü âíósðè îáúåêsà");
		}
		min = initPoint.distance2To(tmpPoint);
		outPoint = tmpPoint;

		bool isIntersecChild = false;
		//for (std::list<Shape *>::iterator i = childs.begin(); i != childs.end(); i++){
		for (int j = 0; j < size; j++){
			Shape * i = childs[j];
			if (i->getIntersectionPointInShape(initPoint, direction, &tmpPoint)){
				MetricUnit dist = initPoint.distance2To(tmpPoint);
				if (dist < min){
					isIntersecChild = true;
					min = dist;
					outShape = i;
					outPoint = tmpPoint;
				}
			}
		}

		*resPoint = outPoint;
		if (isIntersecChild){
			*resShape = outShape->getContainShape(outPoint);
			// Если точка outPoint находится за пределы объкта this
			if (nullptr == *resShape){
				*resShape = parent->getContainParentShape(outPoint)->getContainShape(outPoint);
			}
		} else {
			*resShape = parent->getContainParentShape(outPoint)->getContainShape(outPoint);
		}
		
	}

	__device__ Shape * getParent() const{
		return parent;
	}
	__device__ Shape** getChilds() const{
		return childs;
	}

	__device__ __host__ void setMaterial(const Material * material){
		this->material = material;
	}

	__device__ const Material * getMaterial() const{
		return material;
	}

	__device__ __host__ void addChild(Shape * child){
		//childs.push_back(child);
		childs[size] = child;
		size++;
		child->parent = (Shape *)this;
	}

	/*__device__ __host__ virtual std::string ToString(){
		return "";
	}*/

	Shape * parent;
	//std::list<Shape *> childs;
	Shape ** childs;
	int size;
private:
	
	const Material * material;
};

/*__device__ __host__ Point3D getNearestPoint(Point3D from, Point3D p1, Point3D p2){
	return from.distance2To(p1) < from.distance2To(p2) ? p1 : p2;
}*/

#endif