#ifndef __MTNG_PARTICLE_SOURCE_H__
#define __MTNG_PARTICLE_SOURCE_H__

#define _USE_MATH_DEFINES

#include "Point3D.h"
#include "Vector3D.h"
#include "Particle.h"
#include <math.h>
#include <limits>


#define DOUBLE_EPSOLON 0.00000000001

double randDouble();

class ParticleSource{
public:
	__host__ __device__ ParticleSource(){}
	__host__ __device__ ~ParticleSource(){}
	__device__ virtual Particle generate() = 0;
};

class SimpleParticleSource: public ParticleSource{
public:
	SimpleParticleSource(){}
	~SimpleParticleSource(){}

	__device__ Particle generate(){
		//return Particle(Point3D(0, 0, 0), Vector3D(1, 0, 0), 0.662, ParticleType::GAMMA);
		/*Px = -sin θ cos φ
		Py = -sin θ sin φ
		Pz = -cos θ*/
		double x = randDouble();
		double y = randDouble();
		double z = randDouble();
		Vector3D dir(x, y, z);
		dir.normalize();
		return Particle(Point3D(0, 0, 0), dir, 0.662, ParticleType::GAMMA);
	}

	
};

class IsotropicParticleSource : public ParticleSource{
public:
	IsotropicParticleSource() : pos(Point3D(0, 0, 0)){
	}
	IsotropicParticleSource(Point3D pos) : pos(pos){}
	~IsotropicParticleSource(){}
	
	__device__ Particle generate(){
		/*Px = -sin θ cos φ
		Py = -sin θ sin φ
		Pz = -cos θ*/

		double cosTheta = -1.0 + 2.0*randDouble();
		double phi = 2 * M_PI * randDouble();
		double sinTheta = sqrt(1 - cosTheta*cosTheta);
		double cosPhi = cos(phi);
		double sinPhi = sin(phi);

		/*while (abs(abs(sinTheta) - 1) < std::numeric_limits<double>::epsilon()
			|| abs(abs(sin(phi)) - 1) < std::numeric_limits<double>::epsilon()
			|| abs(abs(cosTheta) - 1) < std::numeric_limits<double>::epsilon()
			|| abs(abs(cos(phi)) - 1) < std::numeric_limits<double>::epsilon()){*/

		while (abs(abs(sinTheta) - 1) < DOUBLE_EPSOLON
			|| abs(abs(sin(phi)) - 1) < DOUBLE_EPSOLON
			|| abs(abs(cosTheta) - 1) < DOUBLE_EPSOLON
			|| abs(abs(cos(phi)) - 1) < DOUBLE_EPSOLON){

			cosTheta = -1.0 + 2.0*randDouble();
			phi = 2 * M_PI * randDouble();
			sinTheta = sqrt(1 - cosTheta*cosTheta);
			cosPhi = cos(phi);
			sinPhi = sin(phi);
		}
		Vector3D dir(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
		/*x += dir.x;
		y += dir.y;
		z += dir.z;*/
		return Particle(pos, dir, 0.662, ParticleType::GAMMA);
	}
private:
	Point3D pos;
};

class SolidAngleParticleSource : public ParticleSource{
public:
	SolidAngleParticleSource(double angle){
		/*if (-M_PI / 2 > angle || M_PI / 2 < angle){
			throw new std::exception("Angle should be between -PI /2 to PI / 2");
		}*/
		this->angle = angle;
	}
	~SolidAngleParticleSource(){}

	__device__ Particle generate(){
		double cosTheta = -1.0 + 2.0*randDouble();
		double phi = 2 * M_PI * randDouble();
		double sinTheta = sqrt(1 - cosTheta*cosTheta);
		double cosPhi = cos(phi);
		double sinPhi = sin(phi);

		while (abs(abs(sinTheta) - 1) < DOUBLE_EPSOLON
			|| abs(abs(sin(phi)) - 1) < DOUBLE_EPSOLON
			|| abs(abs(cosTheta) - 1) < DOUBLE_EPSOLON
			|| abs(abs(cos(phi)) - 1) < DOUBLE_EPSOLON){

			cosTheta = -1.0 + 2.0*randDouble();
			phi = 2 * M_PI * randDouble();
			sinTheta = sqrt(1 - cosTheta*cosTheta);
			cosPhi = cos(phi);
			sinPhi = sin(phi);
		}
		Vector3D dir(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
		//Vector3D dir(sinTheta * cosTheta, sinTheta * sinPhi, cosTheta);
		return Particle(Point3D(0, 0, 0), dir, 0.662, ParticleType::GAMMA);
	}
private:
	double angle;
};


class MonoDirectionalParticleSource : public ParticleSource{
public:
	__host__ __device__ MonoDirectionalParticleSource(Point3D position, Vector3D direction,
		double energy, ParticleType type) : position(position),
		direction(direction), energy(energy), type(type){
	}
	__host__ __device__ ~MonoDirectionalParticleSource(){}

	__device__ Particle generate(){
		return Particle(position, direction, energy, type);
	}

private:
	const double energy;
	const Point3D position;
	const Vector3D direction;
	const ParticleType type;
};

class DoubleDirectionalParticleSource : public ParticleSource{
public:
	DoubleDirectionalParticleSource(Point3D position, Vector3D direction,
		double energy, ParticleType type) : position(position),
		direction(direction), energy(energy), type(type){
	}
	~DoubleDirectionalParticleSource(){}

	__device__ Particle generate(){
		isEven = !isEven;
		return Particle(position, isEven ? direction * -1: direction, energy, type);
	}

private:
	bool isEven = true;
	const double energy;
	const Point3D position;
	const Vector3D direction;
	const ParticleType type;
};

inline double randDouble(){
	int d = rand();
	return (double)(d) / (RAND_MAX);
}

#endif
