#ifndef __MTNG_SIMPLE_PHYSICS_H__
#define __MTNG_SIMPLE_PHYSICS_H__

#define _USE_MATH_DEFINES

#include "IPhysics.h"
#include "Material.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <limits>

#define DOUBLE_EPSOLON 0.00000000001

class SimplePhysics : public IPhysics{

	// photo + comp + pair
	__device__ bool interaction(const Material* material, Particle *particle, InteractionType * type, curandState_t * state) const {
		//   |-----c------|-----p-----|------2--------|
		double rand = randDouble(state);
		double comp = material->getGSigmaCompt(particle->getEnergy()) / material->getGSigmaTotal(particle->getEnergy());
		double compAndPhoto = (material->getGSigmaCompt(particle->getEnergy()) + material->getGSigmaPhoto(particle->getEnergy()))
			/ material->getGSigmaTotal(particle->getEnergy());
		if (comp > rand){
			*type = InteractionType::COMPTON;
			double newEnergy = 0;
			double nu = 0;
			double sinAzim;
			double cosAzim;
			//double 
			kahnAlgoritm(particle->getEnergy(), &newEnergy, state);
			nu = 1 + 0.511 / particle->getEnergy() - 0.511 / newEnergy;
			if (nu > 1){
				nu = 1;
			}
			if (nu < -1){
				nu = -1;
			}
			randAngle(&cosAzim, &sinAzim, state);
			Vector3D newVector = rotateVector(particle->getDirection(), nu, cosAzim, sinAzim, state);
			newVector.normalize();
			while (abs(abs(newVector.z) - 1) < DOUBLE_EPSOLON){
				randAngle(&cosAzim, &sinAzim, state);
				newVector = rotateVector(particle->getDirection(), nu, cosAzim, sinAzim, state);
				newVector.normalize();
			}
			//newVector.normalize();
			particle->setDirection(newVector);
			particle->setEnergy(newEnergy);
			return true;
		} else if (rand < compAndPhoto){
			*type = InteractionType::PHOTO;
			return false;
		} else {
			// if pair
			*type = InteractionType::PAIR;
			return false;
		}
	}

	__device__ MetricUnit calcUniversalFreePath(curandState_t * state) const {
		double d = -log(randDoubleWithoutNull(state));
		return d;
	}

	__device__ MetricUnit calcFreePath(MetricUnit universalFreePath, const Material* material, const Particle &particle) const {
		return (universalFreePath) / material->getGSigmaTotal(particle.getEnergy());
	}

	__device__ void kahnAlgoritm(const double inEnergy, double * outEnergy, curandState_t * state) const{
		double psi1 = 0;
		double psi2 = 0;
		double psi3 = 0;
		double alpha = inEnergy / 0.511;
		double x = 0;
		while (true){
			x = 0;
			psi1 = randDouble(state);
			psi2 = randDouble(state);
			psi3 = randDouble(state);
			if (psi1 <= (1 + 2 * alpha) / (9 + 2 * alpha)){
				x = 1 + 2 * alpha * psi2;
				if (!(psi3 > 4 * (1 / x - 1 / (x * x)))){
					*outEnergy = inEnergy / x;
					return;
				}
			}
			else{
				x = (1 + 2 * alpha) / (1 + 2 * alpha * psi2);
				if (!(psi3 > ((1 - x + alpha) * (1 - x + alpha) / (2 * alpha * alpha) + 1 / (2 * x)))){
					*outEnergy = inEnergy / x;
					return;
				}
			}
		}
	}

	__device__ inline double randDoubleWithoutNull(curandState_t * state) const{
		double d;
		do{
			d = curand_uniform(state);
		} while (0 == d);
		return d;
		//int d = rand();
		//return (double)(d + 1) / ((long)RAND_MAX + 1);
	}

	__device__ inline double randDouble(curandState_t * state) const{
		return curand_uniform(state);
		//int d = rand();
		//return (double)(d) / ((long)RAND_MAX + 1);
	}

	__device__ Vector3D randIsotropicAngle(curandState_t * state) const{
		double cosTheta = -1.0 + 2.0*randDouble(state);
		// TODO PI
		double phi = 2 * 3.1415 * randDouble(state);
		double sinTheta = sqrt(1 - cosTheta*cosTheta);
		if (isnan(sinTheta * cos(phi)) || isnan(sinTheta * sin(phi)) || isnan(cosTheta)){
			phi = phi;
		}
		return Vector3D(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
	}
	 
	// Метод Фон-Неймана
	__device__ void randAngle2(double * cosOut, double * sinOut, curandState_t * state) const{
		double psi1 = randDouble(state);
		double psi2 = randDouble(state);
		double psi3 = randDouble(state);
		double sign = -1;
		if (psi3 > 0.5){
			sign = 1;
		}
		while (psi1 * psi1 + psi2 * psi2 > 1){
			psi1 = randDouble(state);
			psi2 = randDoubleWithoutNull(state);
			// psi1 = 0 && psi1 = 0 => cos & sin = #IND
		}
		*cosOut = (psi1 * psi1 - psi2 * psi2) / (psi1 * psi1 + psi2 * psi2);
		*sinOut = 2 * psi1 * psi2 / (psi1 * psi1 + psi2 * psi2) * sign;
	}

	__device__ void randAngle(double * cosOut, double * sinOut, curandState_t * state) const{
		double psi1 = 2 * M_PI * randDouble(state);
		*cosOut = cos(psi1);
		//*cosOut = -1.0 + 2.0*randDouble();
		//*sinOut = sqrt(1 - *cosOut * *cosOut);
		*sinOut = sin(psi1);
	}

	/*
	Rotate vector (see Panin book, p. 136)
	Добавлен костыль
	*/
	__device__ Vector3D rotateVector(Vector3D cur, double nu, double cosAzim, double sinAzim, curandState_t * state) const{
		const double border_z = 1 - 0.0000000000001;
		double curz = cur.z;
		//Добавлен костыль
		if (border_z <= curz || -border_z >= curz){
			//curz = border_z;
			if (abs(cur.x) + abs(cur.y) > 0.0000001){
				cur.x = cur.x;
			}
			double a = M_PI / 90 / 2000;
			double phi = randDouble(state) * M_PI * 2;
			curz = curz / abs(curz) * cos(a);
			cur.x = sin(a) * cos(phi);
			cur.y = sin(a) * sin(phi);
		}
		/*if (-border_z >= curz){
			curz = -border_z;
		}*/
		

		double tmp = sqrt((1 - nu * nu) / (1 - curz * curz));
		double x = tmp * (cur.x * cur.z * cosAzim - cur.y * sinAzim) + cur.x * nu;
		double y = tmp * (cur.y * cur.z * cosAzim + cur.x * sinAzim) + cur.y * nu;
		double z = -1 * sqrt((1 - nu * nu) * (1 - curz * curz)) * cosAzim + curz * nu;
		if (isnan(x) || isnan(y) || isnan(z)){
			x = x;
		}
		return Vector3D(x, y, z);
	}

	/*__device__ Vector3D rotateVector2(Vector3D cur, double nu, double cosAzim, double sinAzim) const{
		Vector3D newCur(cur.z, cur.x, cur.y);
		Vector3D res = rotateVector(newCur, nu, cosAzim, sinAzim);
		return Vector3D(res.y, res.z, res.x);
	}*/

private:
	double * distributionX;
	double * distributionY;
};

#endif