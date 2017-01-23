#ifndef __MTNG_DETECTOR_ACTION_LISTENER_H__
#define __MTNG_DETECTOR_ACTION_LISTENER_H__

#include "IActions.h"
#include "Point3D.h"

#include <math.h>
#include <iostream>

class DetectorListener : public IActions{

public:
	__device__ DetectorListener(Shape * detector) : detector(detector){}

	__device__ void newParticle(const Particle &particle, long long num_particle){
		isAlreadyDetected = false;
	}

	__device__ void newStep(const Step & step, long long num_particle){
		if (!isAlreadyDetected
			&& detector == step.shapeAfter){

			isAlreadyDetected = true;
			count++;
		}
	}

	void normalizeAndSave(int partCount){
		std::cout << ((double)count) / partCount << std::endl;
	}

	int count = 0;
private:
	Shape * detector = nullptr;
	
	bool isAlreadyDetected = false;
};



class InteractionListener : public IActions{

public:
	__device__ InteractionListener() {
		for (int i = 0; i < 200; i++){
			count[i] = 0;
		}
	}

	__device__ void newParticle(const Particle &particle, long long num_particle){
		oneCount = 0;
	}

	__device__ void newStep(const Step & step, long long num_particle){
		if (!step.isTransition){
			oneCount++;
		}
		if (step.interaction == InteractionType::PHOTO){
			if (oneCount < 200){
				count[oneCount]++;;
			}
		}
	}

	void normalizeAndSave(int partCount){
		for (int i = 0; i < 200; i++){
			std::cout << ((double)count[i]) / partCount << std::endl;
		}
	}

private:
	int count[200];
	int oneCount;
};

#endif
