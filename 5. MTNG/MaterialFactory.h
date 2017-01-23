#ifndef __MTNG_MATERIAL_FACTORY_H__
#define __MTNG_MATERIAL_FACTORY_H__

#include "Material.h"
#include <list>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>


class MaterialFactory{
public:
	enum MaterialNames{
		WATER,
		SiO2,
		H,
		BLACK,
		NaI,
	};

	static MaterialFactory * getInstance(){
		if (MaterialFactory::instance == nullptr){
			MaterialFactory::instance = new MaterialFactory();
		}
		return MaterialFactory::instance;
		//return new MaterialFactory();
	}
	const Material * getMaterialByName(MaterialNames materialName){
		auto it = materials.find(materialName);
		if (it != materials.end()){
			return (*it).second;
		} else {
			std::cout << "Material is not found";
			return nullptr;
		}
	}
	~MaterialFactory();

	const Material * mixMaterials(MaterialNames name1, double weight1, 
		MaterialNames name2, double weight2){
		const Material * mat1 = getMaterialByName(name1);
		const Material * mat2 = getMaterialByName(name2);
		if (nullptr == mat1 || nullptr == mat2){
			return nullptr;
		}
		double weight1Norm = weight1 / (weight1 + weight2);
		double weight2Norm = weight2 / (weight1 + weight2);

		int size;

		const Material * smallSizeMath = mat1;
		if (mat1->size > mat2->size){
			size = mat2->size;
			smallSizeMath = mat2;
		} else {
			size = mat1->size;
		}

		MetricUnit * energies = new MetricUnit[size];
		MetricUnit * total = new MetricUnit[size];
		MetricUnit * photo = new MetricUnit[size];
		MetricUnit * comp = new MetricUnit[size];
		MetricUnit * rayleigh = new MetricUnit[size];

		for (int i = 0; i < size; i++){
			double energy = smallSizeMath->energies[i];
			energies[i] = energy;
			photo[i] = mat1->getGSigmaPhoto(energy) * weight1Norm + mat2->getGSigmaPhoto(energy) * weight2Norm;
			comp[i] = mat1->getGSigmaCompt(energy) * weight1Norm + mat2->getGSigmaCompt(energy) * weight2Norm;
			//rayleigh[i] = mat1->getGSigmaRayleigh(energy) * weight1Norm + mat2->getGSigmaRayleigh(energy) * weight2Norm;
			total[i] = photo[i] + comp[i];// +rayleigh[i];
		}
		return new Material(energies, photo, comp, 0, rayleigh, total, size);
	}
private:
	MaterialFactory(){
/*		readMaterial("C:\\H2O.txt", 1, MaterialNames::WATER);
		readMaterial("C:\\SiO2.txt", 2.65, MaterialNames::SiO2);
		//readMaterial("C:\\H.txt", 0.0000899, MaterialNames::H);
		readMaterial("C:\\Black.txt", 10000, MaterialNames::BLACK);
		readMaterial("C:\\NaI.txt", 3.67, MaterialNames::NaI);
		*/
		readMaterial("materials/H2O.txt", 1, MaterialNames::WATER);
		readMaterial("materials/SiO2.txt", 2.65, MaterialNames::SiO2);
		//readMaterial("materials/H.txt", 0.0000899, MaterialNames::H);
		readMaterial("materials/Black.txt", 10000, MaterialNames::BLACK);
		readMaterial("materials/NaI.txt", 3.67, MaterialNames::NaI);
		
	}
	void readMaterial(std::string fileName, double nu, MaterialNames name){
		std::fstream file(fileName, std::ios::in);
		if (!file.is_open()){
			std::cout << "Not found material file: " << fileName << std::endl;
			return;
		}

		std::vector<MetricUnit> energiesList;
		std::vector<MetricUnit> totalList;
		std::vector<MetricUnit> photoList;
		std::vector<MetricUnit> compList;
		std::vector<MetricUnit> raylaighList;
		std::vector<MetricUnit> pairList;
		
		double tmp;
		double energy;
		double compVal;
		double photoVal;
		double rayleightVal;
		double pairVal;

		try{
			while (!file.eof()){
				file >> energy;
				energiesList.push_back(energy);
				file >> rayleightVal;
				raylaighList.push_back(rayleightVal * nu);
				file >> compVal;
				compList.push_back(compVal * nu);
				file >> photoVal;
				photoList.push_back(photoVal * nu);
				file >> pairVal;
				pairList.push_back(pairVal * nu);
				file >> tmp;
				file >> tmp;
				file >> tmp;
				totalList.push_back(nu * (compVal + photoVal + pairVal));
				//totalList.push_back(nu * (compVal + photoVal));
			}
		}
		catch (std::exception e){}
		file.close();
		int size = energiesList.size();

		MetricUnit * energies = new MetricUnit[size];
		MetricUnit * total = new MetricUnit[size];
		MetricUnit * photo = new MetricUnit[size];
		MetricUnit * comp = new MetricUnit[size];
		MetricUnit * rayleigh = new MetricUnit[size];

		for (int i = 0; i < size; i++){
			energies[i] = energiesList.at(i);
			total[i] = totalList.at(i);
			photo[i] = photoList.at(i);
			comp[i] = compList.at(i);
			rayleigh[i] = raylaighList.at(i);
		}
		materials.insert(std::pair<MaterialNames, Material *>(name, new Material(energies, photo, comp, 0, rayleigh, total, size)));
	}

private:
	static MaterialFactory * instance;
	//MaterialFactory * instance;
	std::map<MaterialNames, const Material *> materials;
	Material * h2o;
};

#endif
