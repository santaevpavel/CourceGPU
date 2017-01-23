#ifndef __MTNG_PSYSICS_H__
#define __MTNG_PSYSICS_H__

#include "Particle.h"
#include "Material.h"
#include <curand.h>
#include <curand_kernel.h>
#include <list>

enum InteractionType{
	TRANSITION,
	PHOTO,
	COMPTON,
	RAYLAEIGH,
	PAIR,
};

class IPhysics{
public:

public:
	__device__ virtual bool interaction(const Material* material, Particle *particle, InteractionType * type, curandState_t * state) const = 0;

	__device__ virtual MetricUnit calcUniversalFreePath(curandState_t * state) const = 0;

	__device__ virtual MetricUnit calcFreePath(MetricUnit universalFreePath, const Material* material, const Particle &particle) const = 0;
};



#endif