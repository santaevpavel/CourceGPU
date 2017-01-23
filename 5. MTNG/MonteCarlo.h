#ifndef __MTNG_MONTE_CARLO_H__
#define __MTNG_MONTE_CARLO_H__

#include "Point3D.h"
#include "Vector3D.h"
#include "Util.h"
#include <curand.h>
#include <curand_kernel.h>

#include "ParticleSource.h"
#include "IPhysics.h"
#include "IActions.h"
#include "DetectorListener.h"

#include <queue>
#include <list>
#include <iostream>

class MonteCarlo{
public:

	__device__ __host__ MonteCarlo(ParticleSource * source, const IPhysics * physics, 
		const Scene * scene, IActions ** actions, int size)
		: source(source), physics(physics), scene(scene){
		//this->actions.splice(this->actions.begin(), actions);

		this->actions = actions;
		this->actionSize = size;
	}

	__device__ __host__ MonteCarlo(ParticleSource * source, const IPhysics * physics,
		const Scene * scene)
		: source(source), physics(physics), scene(scene){}

	__device__ void run(long particleCount){

		curandState_t state;
		curand_init(0, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

		for (long i = 0; i < particleCount; i++){
			((DetectorListener * )actions[0])->count++;
			Particle particle = source->generate();
			//for (auto it = actions.cbegin(); it != actions.cend(); it++){
			for (int j = 0; j < actionSize; j++){
				(actions[j])->newParticle(particle, i);
			}
			runOneTrack(particle, i, state);
		}
	}

	/*__device__ void addAction(IActions * action){
		actions.push_back(action);
	}*/

private:
	__device__ void runOneTrack(const Particle& particle, long i, curandState_t state){
		//return;
		Shape * nextShape;
		Point3D nextPoint;
		Point3D curPoint = particle.getPosition();
		MetricUnit unversalFreePath;
		MetricUnit meanFreePath;
		bool particleIsAlive = true;
		bool isChangeShape = false;
		InteractionType interactionType = InteractionType::PHOTO;
		Particle oldParticle = particle;
		Shape* curShape = NULL;
		Particle curParticle = particle;
		curShape = scene->root->childs[0];//scene->getContainsShape(particle.getPosition());
		particleIsAlive = true;
		int k = 0;
		return;
		do{
			k++;
			if (!isChangeShape){
				unversalFreePath = physics->calcUniversalFreePath(&state);
			} else {
				isChangeShape = false;
			}
			meanFreePath = physics->calcFreePath(unversalFreePath, curShape->getMaterial(), curParticle);				
			curShape->getNearIntersectionPoint(curParticle.getPosition(), curParticle.getDirection(), 0, &nextPoint, &nextShape);
			if ((nextPoint - curParticle.getPosition()).length() < meanFreePath){
				oldParticle = curParticle;
				unversalFreePath = unversalFreePath - unversalFreePath  * ((nextPoint - curParticle.getPosition()).length()) / meanFreePath;
				curParticle.setPosition(nextPoint);
				//for (auto it = actions.cbegin(); it != actions.cend(); it++){
				for (int j = 0; j < actionSize; j++){
					Step step(oldParticle, curParticle, curShape, nextShape, true, TRANSITION);
					(actions[j])->newStep(step, i);
				}
				curPoint = nextPoint;
				curShape = nextShape;
				isChangeShape = true;
			} else {
				curPoint = curPoint + curParticle.getDirection() * meanFreePath;
				oldParticle = curParticle;
				curParticle.setPosition(curPoint);
				particleIsAlive = physics->interaction(curShape->getMaterial(), &curParticle, &interactionType, &state);
				//for (auto it = actions.cbegin(); it != actions.cend(); it++){
				for (int j = 0; j < actionSize; j++){
					Step step(oldParticle, curParticle, curShape, curShape, false, interactionType);
					(actions[j])->newStep(step, i);
				}
			}
		} while (particleIsAlive);
		//}
		return;
	}
private:
	ParticleSource * source;
	const IPhysics * physics;
	//std::list<IActions *> actions;
	IActions ** actions;
	int actionSize;
	const Scene * scene;
};

#endif