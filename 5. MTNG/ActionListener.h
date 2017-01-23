#ifndef __MTNG_ACTION_LISTENER_H__
#define __MTNG_ACTION_LISTENER_H__

#include "IActions.h"
#include "Point3D.h"

#include <math.h>
#include <iostream>

#define SPHERE_RADIUS 50

class ActionListener : public IActions{
public:
	double counts[SPHERE_RADIUS];
	double dist = 0;

	__device__ ActionListener(){
		for (int i = 0; i < SPHERE_RADIUS; i++){
			counts[i] = 0;
		}
	}
	__device__ void newParticle(const Particle &particle, long long num_particle){
		/*std::cout << "New " << num_particle << " ";
		printParticle(particle);
		std::cout << "\n";
		dist = 0;*/
	}
	__device__ void newStep(const Step & step, long long num_particle){
		//double d = step.particleAfter.getPosition().distanceTo(step.particleBefore.getPosition());
		//bool isTransition = step.shapeAfter != step.shapeBefore;
		//if (isTransition){
		double energy = step.particleBefore.getEnergy();
		/*if (energy > 0.2){
			return;
		}*/
			int i = (int)step.particleAfter.getPosition().distanceTo(Point3D(0, 0, 0));
			int j = (int)step.particleBefore.getPosition().distanceTo(Point3D(0, 0, 0));
			if (i < j){
				int k = i;
				i = j;
				j = k;
			}
			for (int x = j + 1; x <= i; x++){
				if (0 <= x && x < SPHERE_RADIUS){
					counts[x] = counts[x] + 1;
				}
			}
		//}
		/*std::cout << "Step  " << isTransition << "\n";
		printParticle(step.particleBefore);
		//printParticle(step.particleAfter);
		std::cout << "\n";
		//dist += d;*/
	}

	/*__device__ void printParticle(Particle p){
		Point3D pos = p.getPosition();
		Vector3D dir = p.getDirection();
		std::cout << "\tpos: " << pos.x << " " << pos.y << " " << pos.z << "\n\tdir: "
			<< dir.x << " " << dir.y << " " << dir.z << "\n";

	}*/

	void normalizeAndSave(int count){
		for (int i = 0; i < SPHERE_RADIUS; i++){
			counts[i] = counts[i] / (4 * 3.14159265358979 * i * i) / count;
		}
		for (int i = 0; i < SPHERE_RADIUS; i++){
			std::cout << counts[i] << "\n";
		}
	}
};

#endif
