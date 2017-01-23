/*#ifndef __MTNG_FREE_PATH_ACTION_LISTENER_H__
#define __MTNG_FREE_PATH_ACTION_LISTENER_H__

#include "IActions.h"
#include "Point3D.h"
#include "MaterialFactory.h"

#include <math.h>
#include <iostream>

class FreePathActionListener : public IActions{
public:
	long count = 0;
	long double dist = 0;
	const Material * mat;
	long compCount = 0;
	long relCount = 0;
	long photCount = 0;
	long sumCount = 0;
	long double relDistSum = 0;
	bool isRel = false;

	FreePathActionListener(){
		mat = MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::WATER);
	}

	void newParticle(const Particle &particle, long long num_particle){
	}
	void newStep(const Step & step, long long num_particle){
		bool isTransition = step.shapeAfter != step.shapeBefore;
		double energy = step.particleBefore.getEnergy();
		/*if (0.2 < energy){
			return;
		}*/
		if (!step.isTransition){
			count++;
			sumCount++;
			if (isRel){
				relDistSum += step.particleAfter.getPosition().distanceTo(step.particleBefore.getPosition());
			}
			isRel = false;
			if (InteractionType::COMPTON == step.interaction){
				compCount++;
			}
			else if (InteractionType::RAYLAEIGH == step.interaction){
				relCount++;
				isRel = true;
			}
			else if (InteractionType::PHOTO == step.interaction){
				photCount++;
			}
		}
		
		dist += step.particleAfter.getPosition().distanceTo(step.particleBefore.getPosition());


		//std::cout << step.particleBefore.getEnergy() << "\t" << step.particleAfter.getPosition().distanceTo(step.particleBefore.getPosition()) <<
		//	"\t" << mat->getGSigmaTotal(step.particleBefore.getEnergy()) << "\n";
	}

	void printParticle(Particle p){
		Point3D pos = p.getPosition();
		Vector3D dir = p.getDirection();
		std::cout << "\tpos: " << pos.x << " " << pos.y << " " << pos.z << "\n\tdir: "
			<< dir.x << " " << dir.y << " " << dir.z << "\n";

	}

	void printResults(int partCount){
		std::cout << "1) Full distance\n2) Full counts\n3) Avarage distance \n5) Avatage counts\n" << std::endl;
		std::cout << dist << std::endl;
		std::cout << count << std::endl;
		std::cout << dist / partCount << std::endl;
		std::cout << ((double)count) / partCount << std::endl;
		std::cout << "Sum, phot, rel, comp" << std::endl;
		std::cout << sumCount << std::endl;
		std::cout << photCount << std::endl;
		std::cout << relCount << " dist " << relDistSum / relCount << std::endl;
		std::cout << compCount << std::endl;
	}
};

#endif


*/