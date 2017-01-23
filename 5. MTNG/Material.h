#ifndef __MTNG_MATERIAL_H__
#define __MTNG_MATERIAL_H__

#include <iostream>
#include "Util.h"

// Ётот класс с точки зрени€ ќќјиƒ надо переделать, мы должны передавать частицу и получать тип взаимодейсти€ или возможно по другому???
class Material{
public:
	__device__ __host__ Material(MetricUnit * energies, MetricUnit * sigmaPhoto,
		MetricUnit * sigmaCompt, MetricUnit * sigmaPair,
		MetricUnit * sigmaRayleigh, MetricUnit * sigmaTotal, int size) : 
		energies(energies), sigmaPhoto(sigmaPhoto),
		sigmaCompt(sigmaCompt), sigmaPair(sigmaPair),
		sigmaRayleigh(sigmaRayleigh), sigmaTotal(sigmaTotal), size(size){
	}

	__device__ __host__ MetricUnit getGSigmaPhoto(double energy) const{
		int i = findEnergy(energy);
		if (i == 0){
			return sigmaPhoto[i];
		}
		double s1 = sigmaPhoto[i - 1]; 
		double s2 = sigmaPhoto[i];
		double e1 = energies[i - 1];
		double e2 = energies[i];
		return linerInterpolate(e1, s1, e2, s2, energy);
	}

	__device__ __host__ MetricUnit getGSigmaRayleigh(double energy) const{
		int i = findEnergy(energy);
		if (i == 0){
			return sigmaRayleigh[i];
		}
		double s1 = sigmaRayleigh[i - 1];
		double s2 = sigmaRayleigh[i];
		double e1 = energies[i - 1];
		double e2 = energies[i];
		return linerInterpolate(e1, s1, e2, s2, energy);
	}

	__device__ __host__  MetricUnit getGSigmaCompt(double energy) const{
		int i = findEnergy(energy);
		if (i == 0){
			return sigmaCompt[i];
		}
		double s1 = sigmaCompt[i - 1];
		double s2 = sigmaCompt[i];
		double e1 = energies[i - 1];
		double e2 = energies[i];
		return linerInterpolate(e1, s1, e2, s2, energy);
	}

	__device__ __host__ MetricUnit getGSigmaPair(double energy) const{
		int i = findEnergy(energy);
		if (i == 0){
			return sigmaPair[i];
		}
		double s1 = sigmaPair[i - 1];
		double s2 = sigmaPair[i];
		double e1 = energies[i - 1];
		double e2 = energies[i];
		return linerInterpolate(e1, s1, e2, s2, energy);
	}
	__device__ __host__ MetricUnit getGSigmaTotal(double energy) const{
		int i = findEnergy(energy);
		if (i == 0){
			return sigmaTotal[i];
		}
		double s1 = sigmaTotal[i - 1];
		double s2 = sigmaTotal[i];
		double e1 = energies[i - 1];
		double e2 = energies[i];
		return linerInterpolate(e1, s1, e2, s2, energy);
	}

	/*__device__ double getFormFactor(double q){

	}*/

	friend class MaterialFactory;

	MetricUnit * energies;
	MetricUnit * sigmaPhoto; //массив в котором хран€тс€ сечени€, загружены из файла
	MetricUnit * sigmaCompt;
	MetricUnit * sigmaPair;
	MetricUnit * sigmaTotal;
	MetricUnit * sigmaRayleigh;
	int size;

private:
	
	__device__ __host__ int findEnergy(double energy) const {
		if (energy < energies[0]){
			return 0;
		}
		for (int i = 0; energies[i + 1] != 0; i++){
			if (energies[i] <= energy && energy <= energies[i + 1]){
				return i + 1;
			}
		}
		//std::cout << "Error: not found energy " << energy;
		return 0;
	}

	__device__ __host__ inline MetricUnit linerInterpolate(MetricUnit x0, MetricUnit y0, MetricUnit x1, MetricUnit y1, MetricUnit x) const{
		/*double _y0 = log(y0);
		double _y1 = log(y1);
		double _x0 = log(x0);
		double _x1 = log(x1);
		double _x = log(x);*/
		//return exp(_y0 * (x - x1) / (x0 - x1) + _y1 * (x - x0) / (x1 - x0));
		return y0 * (x - x1) / (x0 - x1) + y1 * (x - x0) / (x1 - x0);
		//return (y0+y1)/2;
	}


};
#endif