#ifndef __MTNG_STEP_H__
#define __MTNG_STEP_H__

#include "Shape.h"
#include "Particle.h"
#include "IPhysics.h"

class Step{
public:
	__device__ __host__ Step();
	__device__ __host__ Step(const Particle particleBefore, const Particle particleAfter, const Shape * shapeBefore,
		const Shape * shapeAfter, const bool isTransition, const InteractionType interaction) :
		particleBefore(particleBefore), particleAfter(particleAfter),
		shapeBefore(shapeBefore), shapeAfter(shapeAfter), isTransition(isTransition), interaction(interaction){
	}
	const Particle particleBefore;
	const Particle particleAfter;
	const Shape * shapeBefore;
	const Shape * shapeAfter;
	const bool isTransition;
	const InteractionType interaction;
};

#endif