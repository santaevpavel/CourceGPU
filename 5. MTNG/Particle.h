#ifndef __MTNG_PARTICLE_H__
#define __MTNG_PARTICLE_H__

#include "Point3D.h"
#include "Vector3D.h"
#include "Util.h"

enum ParticleType{
	GAMMA,
	NEUTRON,
};

class Particle{
public:
	__host__ __device__ Particle(){
		Particle(Point3D(0, 0, 0), Vector3D(1, 0, 0), 0, ParticleType::GAMMA);
	}

	__host__ __device__ Particle(const Point3D &pos,const Vector3D &dir,const double energy,const ParticleType &type): position(pos), direction(dir), energy(energy), type(type){
		direction.normalize();
	}
	__device__ Point3D getPosition() const{
		return position;
	}

	__device__ Vector3D getDirection() const{
		return direction;
	}

	__device__ double getEnergy() const{
		return energy;
	}

	__device__ ParticleType getType() const{
		return type;
	}

	__device__ void setPosition(Point3D pos){
		position = pos;
	}

	__device__ void setDirection(Vector3D dir){
		direction = dir;
	}

	__device__ void setEnergy(double energy){
		this->energy = energy;
	}

	__device__ void setType(ParticleType type){
		this->type;
	}
private:
	Point3D position;
	Vector3D direction;
	double energy;
	ParticleType type;
	MetricUnit passedLength;
};



#endif
