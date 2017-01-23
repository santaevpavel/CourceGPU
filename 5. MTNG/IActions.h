#ifndef __MTNG_ACTIONS_H__
#define __MTNG_ACTIONS_H__

#include "Step.h"

//События связанные с расчетом
class IActions{
public:
	__device__ virtual void newParticle(const Particle &particle,long long num_particle){}
	__device__ virtual void newStep(const Step & step, long long num_particle){}
};

#endif