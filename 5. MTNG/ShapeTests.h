#define _USE_MATH_DEFINES

#ifndef __MTNG_SHAPE_TEST_H__
#define __MTNG_SHAPE_TEST_H__

#include <iostream>
#include <time.h>
#include <math.h>
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
#include "MaterialFactory.h"
#include "CrossingListener.h"

using namespace std;

__global__ void kernel(Shape * shape, Sphere * detector, ParticleSource * source, int count,
		Material * water, Material * sio2, int * res){
		
		Sphere * universe = new Sphere(Point3D(0, 0, 0), 1000000);
		Sphere * world = new Sphere(Point3D(0, 0, 0), 100000);
		world->setMaterial(water);
		universe->setMaterial(sio2);

		universe->addChild(world);
		world->addChild(shape);
		world->addChild(detector);

		shape->size = 0;
		detector->size = 0;

		Scene scene(universe);

		IPhysics * physics = new SimplePhysics();
		IActions ** actions = new IActions*[1];
		DetectorListener * action = new DetectorListener(detector);
		actions[0] = action;

		Vector3D dir(1, 0, 0);
		MonoDirectionalParticleSource  * source2 = new MonoDirectionalParticleSource(Point3D(-1, 0, 0), dir, 0.662, ParticleType::GAMMA);
		source = source2;
		MonteCarlo monteCarlo(source, physics, &scene, actions, 1);

		action->count = 5;

		monteCarlo.run(count);

		*res = action->count;
}

class ShapeTest{

public:

	Shape * copyShape(const Shape * host){
		Shape * device;
		cudaMalloc(&device, sizeof(Sphere));
		cudaMemcpy(device, host,
    		sizeof(Sphere), cudaMemcpyHostToDevice);	
		return device;
	}

	Material * copyMaterial(const Material * host){
		Material * device;

		int size = host->size;
		MetricUnit * energies;
		MetricUnit * sigmaPhoto;
		MetricUnit * sigmaCompt;
		MetricUnit * sigmaPair;
		MetricUnit * sigmaTotal;

		cudaMalloc(&device, sizeof(Material));

		cudaMalloc(&energies, sizeof(MetricUnit) * size);
		cudaMalloc(&sigmaPhoto, sizeof(MetricUnit) * size);
		cudaMalloc(&sigmaCompt, sizeof(MetricUnit) * size);
		cudaMalloc(&sigmaPair, sizeof(MetricUnit) * size);
		cudaMalloc(&sigmaTotal, sizeof(MetricUnit) * size);

		cudaMemcpy(energies, host->energies, sizeof(MetricUnit) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(sigmaPhoto, host->sigmaPhoto, sizeof(MetricUnit) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(sigmaCompt, host->sigmaCompt, sizeof(MetricUnit) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(sigmaPair, host->sigmaPair, sizeof(MetricUnit) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(sigmaTotal, host->sigmaTotal, sizeof(MetricUnit) * size, cudaMemcpyHostToDevice);

		Material * mat = new Material(energies, sigmaPhoto, sigmaCompt, sigmaPair, NULL, sigmaTotal, size);

		cudaMemcpy(device, mat,
    		sizeof(Material), cudaMemcpyHostToDevice);	
		return device;
	}

	void test(Shape * shape, Sphere * detector, ParticleSource * source, int count){
		
		const Material * water = MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::WATER);
		const Material * sio2 = MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::SiO2);

		Material * waterDevice = copyMaterial(water);	
		Material * waterSio2 = copyMaterial(sio2);	

		Shape * shapeDevice = copyShape(shape);	
		Sphere * detectorDevice = (Sphere *) copyShape(detector);	

		ParticleSource * sourceDevice;
		cudaMalloc(&sourceDevice, sizeof(MonoDirectionalParticleSource));
		cudaMemcpy(sourceDevice, source,
    		sizeof(MonoDirectionalParticleSource), cudaMemcpyHostToDevice);	

		dim3 threads = dim3(1, 1);
    	dim3 blocks  = dim3(1, 1);

    	int *resDevice;
    	int *resHost = new int[1];
    	cudaMalloc(&resDevice, sizeof(int));
		kernel<<<blocks, threads>>>(shapeDevice, detectorDevice, sourceDevice, count, waterDevice, waterSio2, resDevice);
		cudaMemcpy(resHost, resDevice,
    		sizeof(int), cudaMemcpyDeviceToHost);	
		//std::cout << "Сount = " << count << std::endl;
		std::cout << "Res = " << *resHost << std::endl;
	}

	void test(Shape * shape, int N){
		const int count = 12;
		const double deltaAngle = 2 * M_PI / count;
		const double radius = 20;
		const double detectorRadius = 5;

		double angle = 0;
		double x = 0;
		double y = 0;
		double z = 0;

		// x y
		for (int i = 0; i < count; i++){
			angle = deltaAngle * i;
			x = radius * cos(angle);
			y = radius * sin(angle);
			z = 0;
			Sphere * detector = new Sphere(Point3D(x, y, z), detectorRadius);
			detector->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::WATER));
			//IsotropicParticleSource * source = new IsotropicParticleSource(Point3D(-x, -y, -z));
			Vector3D dir(x, y, z);
			dir.normalize();
			MonoDirectionalParticleSource  * source = new MonoDirectionalParticleSource(Point3D(-x, -y, -z), dir, 0.662, ParticleType::GAMMA);
			test(shape, detector, source, N);
		}
		// y z
		for (int i = 0; i < count; i++){
			angle = deltaAngle * i;
			x = 0;
			y = radius * cos(angle);
			z = radius * sin(angle);
			Sphere * detector = new Sphere(Point3D(x, y, z), detectorRadius);
			detector->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::WATER));
			//IsotropicParticleSource * source = new IsotropicParticleSource(Point3D(-x, -y, -z));
			Vector3D dir(x, y, z);
			dir.normalize();
			MonoDirectionalParticleSource  * source = new MonoDirectionalParticleSource(Point3D(-x, -y, -z), dir, 0.662, ParticleType::GAMMA);
			test(shape, detector, source, N);
		}
		// x z
		for (int i = 0; i < count; i++){
			angle = deltaAngle * i;
			x = radius * cos(angle);
			y = 0;
			z = radius * sin(angle);
			Sphere * detector = new Sphere(Point3D(x, y, z), detectorRadius);
			detector->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::WATER));
			//IsotropicParticleSource * source = new IsotropicParticleSource(Point3D(-x, -y, -z));
			Vector3D dir(x, y, z);
			dir.normalize();
			MonoDirectionalParticleSource  * source = new MonoDirectionalParticleSource(Point3D(-x, -y, -z), dir, 0.662, ParticleType::GAMMA);
			test(shape, detector, source, N);
		}
	}

	void testSphere(int count){
		Sphere * shape = new Sphere(Point3D(0, 0, 0), 5);
		shape->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::SiO2));
		test(shape, count);
	}

	void testCylinder(int count){
		Cylinder * shape = new Cylinder(Point3D(0, 0, 0), 5, 0, 3.0f);
		shape->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::SiO2));
		test(shape, count);
	}

	void testSphereIntersection(int count){
		Sphere * sphere1 = new Sphere(Point3D(2.5, 0, 0), 5);
		Sphere * sphere2 = new Sphere(Point3D(-2.5, 0, 0), 5);

		ShapesIntersection * unionShape = new ShapesIntersection(sphere1, sphere2);
		unionShape->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::SiO2));
		test(unionShape, count);
	}

	void testSphereExclusion(int count){
		Sphere * sphere1 = new Sphere(Point3D(0, 0, 0), 5);
		//Sphere * sphere2 = new Sphere(Point3D(2.5, 0, 0), 5);
		Sphere * sphere2 = new Sphere(Point3D(0, 0, 0), 2.5);

		ShapesExclusion * unionShape = new ShapesExclusion(sphere1, sphere2);
		unionShape->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::SiO2));
		test(unionShape, count);
	}

	void fullTest(int count){
		//testSphereIntersection();
		//testSphereExclusion();
		//testCylinder();
		testSphere(count);
	}

	/*

	void test(Shape * shape, Sphere * detector, ParticleSource * source, int count){
		char c;
		//srand(time(NULL));

		Sphere * universe = new Sphere(Point3D(0, 0, 0), 1000000);
		Sphere * world = new Sphere(Point3D(0, 0, 0), 100000);
		world->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::WATER));
		universe->setMaterial(MaterialFactory::getInstance()->getMaterialByName(MaterialFactory::MaterialNames::WATER));

		universe->addChild(world);
		world->addChild(shape);
		world->addChild(detector);

		Scene scene(universe);

		IPhysics * physics = new SimplePhysics();
		IActions ** actions = new IActions*[1];
		DetectorListener * action = new DetectorListener(detector);
		actions[0] = action;

		MonteCarlo monteCarlo(source, physics, &scene, actions, 1);
		std::ofstream myfile;
		myfile.open("timeMTNG.txt", std::fstream::app);
		const clock_t begin_time = clock();

		//int count = 1000 * 1000;
		//monteCarlo.run(count);

		myfile << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;
		myfile.close();

		action->normalizeAndSave(count);
	}

	*/
};

#endif