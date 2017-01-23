#define _USE_MATH_DEFINES

#ifndef __GGKP_TEST_H__
#define __GGKP_TEST_H__

#include <iostream>
#include <time.h>
#include <math.h>
#include "MaterialFactory.h"
#include "Shape.h"
#include "Sphere.h"
#include "Box.h"
#include "ShapesIntersection.h"
#include "ShapesUnion.h"
#include "ShapesExclusion.h"
#include "Scene.h"
#include "Cylinder.h"
#include "Particle.h"
#include "ParticleSource.h"
#include "MonteCarlo.h"
#include "SimplePhysics.h"
#include "ActionListener.h"
#include "DetectorListener.h"
#include "CrossingListener.h"

using namespace std;

class TestGGKP{

public:
	void printTime(time_t t){
		struct tm * now = localtime(&t);
		cout << (now->tm_hour) << ' '
			<< (now->tm_min) << ' '
			<< now->tm_sec
			<< endl;
	}

	void fullTest(int count, int x, int weightSio2){

		/*for (int i = 0; i < 15; i++){
			for (int w = 0; w < 20; w++){
				test(10.8 - (double)i - 1, (double)(1 - w / 100), count);
			}
		}*/

		test(10.8 - x - 1, ((double)weightSio2) / 100, count);
	}

	void test(double x, double sio2Weight, int count){
		char c;
		//srand(time(NULL));

		/*------------------ SCENE ------------------*/
		Sphere * universe = new Sphere(Point3D(0, 0, 0), 1000000);
		Sphere * world = new Sphere(Point3D(0, 0, 0), 100000);
		//world->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::SiO2));
		universe->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::WATER));

		universe->addChild(world);

		Scene scene(universe);

		IPhysics * physics = new SimplePhysics();
		/*------------------- SHAPE -----------------*/
		Vector3D sourceDir(1, 0, 1);
		sourceDir.normalize();
		MonoDirectionalParticleSource  * source = new MonoDirectionalParticleSource(Point3D(x + 1.001, 0, -0.00001), 
			sourceDir, 0.662, ParticleType::GAMMA);
		
		double detectorHeight = 3.35;
		double cylinder2Height = 10 - 3.35;
		double cylinder2CenterZ = 30 - cylinder2Height / 2;
		double detector2CenterZ = 30 + detectorHeight / 2;
		double cylinder3CenterZ = detector2CenterZ + detectorHeight / 2 + 1;

		Cylinder * skvazina = new Cylinder(Point3D(0, 0, 0), 10.8, 0, 10000);

		Cylinder * cylinder1 = new Cylinder(Point3D(x, 0, 10), 1, 0, 10);
		Shape * cylinderDetector1 = buildDetector(true, skvazina, x); //new Cylinder(Point3D(0, 0, 20 + detectorHeight / 2), 1, 0, detectorHeight / 2);
		Cylinder * cylinder2 = new Cylinder(Point3D(x, 0, cylinder2CenterZ), 1, 0, cylinder2Height / 2);
		Shape * cylinderDetector2 = buildDetector(false, skvazina, x); //new Cylinder(Point3D(0, 0, detector2CenterZ), 1, 0, detectorHeight / 2);
		Cylinder * cylinder3 = new Cylinder(Point3D(x, 0, cylinder3CenterZ), 1, 0, 1);

		world->setMaterial(MaterialFactory::getInstance()->mixMaterials(MaterialFactory::MaterialNames::SiO2, sio2Weight,
			MaterialFactory::MaterialNames::WATER, 1 - sio2Weight));
		//world->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::SiO2));
		skvazina->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::WATER));

		cylinder1->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::BLACK));
		cylinder2->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::BLACK));
		cylinder3->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::BLACK));

		skvazina->addChild(cylinder1);
		skvazina->addChild(cylinder2);
		skvazina->addChild(cylinder3);

		world->addChild(skvazina);

		std::list<IActions *> actions;
		StreamDetectorListener * action1 = new StreamDetectorListener(cylinderDetector1);
		StreamDetectorListener * action2 = new StreamDetectorListener(cylinderDetector2);

		DetectorListener * action1c = new DetectorListener(cylinderDetector1);
		DetectorListener * action2c = new DetectorListener(cylinderDetector2);

		//InteractionListener * interactionListener = new InteractionListener();

		//actions.push_back(interactionListener);

		actions.push_back(action1);
		actions.push_back(action2);

		actions.push_back(action1c);
		actions.push_back(action2c);

		/*----------------- MODELLING ---------------*/
		MonteCarlo monteCarlo(source, physics, &scene, actions);
		//std::ofstream myfile;
		//myfile.open("timeMTNG.txt", std::fstream::app);
		const clock_t begin_time = clock();

		//int count = 1000 * 1000;
		monteCarlo.run(count);

		//myfile << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;
		//myfile.close();

		//std::cout << "Detector 1" << std::endl;
		action1->normalizeAndSaveNoEOL(count);
		std::cout << "\t";
		action1->normalizeAndSaveErrorNoEOL(count);
		std::cout << "\t";
		action2->normalizeAndSaveNoEOL(count);
		std::cout << "\t";
		action2->normalizeAndSaveErrorNoEOL(count);
		std::cout << "\t";

		action1c->normalizeAndSaveNoEOL(count);
		std::cout << "\t";
		action2c->normalizeAndSaveNoEOL(count);
		std::cout << std::endl;

		//interactionListener->normalizeAndSave(count);
	}

	Shape * buildDetector(bool isFirst, Cylinder * skvazina, double x){
		double detectorHeight = 3.35;
		double cylinder2Height = 10 - 3.35;
		double cylinder2CenterZ = 30 - cylinder2Height / 2;
		double detector2CenterZ = 30 + detectorHeight / 2;
		double cylinder3CenterZ = detector2CenterZ + detectorHeight / 2 + 1;

		if (isFirst){
			return buildDetector(detectorHeight / 2, 20 + detectorHeight / 2, skvazina, x);
		} else {
			return buildDetector(detectorHeight / 2, detector2CenterZ, skvazina, x);
		}
	}

	Shape * buildDetector(double halfHeight, double z, Cylinder * skvazina, double x){
		double angle = M_PI / 6;
		Point3D windowPoint(cos(angle / 2), sin(angle / 2), 0);
		/*double xSmall = 0.1;
		double smallDetectorRad = Point3D(xSmall, 0, 0).distanceTo(windowPoint);

		Cylinder * cylinderBig = new Cylinder(Point3D(x, 0, z), 1, 0, halfHeight);
		Cylinder * cylinderSmall = new Cylinder(Point3D(x + xSmall, 0, z), smallDetectorRad, 0, halfHeight);

		cylinderBig->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::BLACK));
		cylinderSmall->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::BLACK));*/


		// С коллиционным окон
		Cylinder * cylinderOutter = new Cylinder(Point3D(x, 0, z), 1, 0, halfHeight);
		Cylinder * cylinderWindow = new Cylinder(Point3D(x + 1, 0, z), sin(angle / 2), 0, halfHeight);
		Cylinder * cylinderInner = new Cylinder(Point3D(x, 0, z), 0.99, 0, halfHeight);
		Cylinder * cylinderInnerExcl = new Cylinder(Point3D(x, 0, z), 0.99, 0, halfHeight);

		ShapesExclusion * outterWithWindow = new ShapesExclusion(cylinderOutter, cylinderWindow);
		ShapesExclusion * out = new ShapesExclusion(outterWithWindow, cylinderInnerExcl);
		out->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::BLACK));
		cylinderInner->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::NaI));

		//out->addChild(cylinderInner);

		skvazina->addChild(out);
		skvazina->addChild(cylinderInner);
		
		// Без коллимационного окна
		/*Cylinder * cylinderInner = new Cylinder(Point3D(x, 0, z), 1, 0, halfHeight);
		cylinderInner->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::NaI));

		skvazina->addChild(cylinderInner);*/

		return cylinderInner;
	}
};

#endif