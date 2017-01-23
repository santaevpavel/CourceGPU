#define _USE_MATH_DEFINES

#include <iostream>
#include <time.h>
#include <math.h>

//#include "TestGGKP.h"
#include "ShapeTests.h"

//using namespace std;

int main(int argc, char ** argv){
	// Range test
	//TwoMevRangeTest test;
	//test.fullTest();
	/*if (argc < 4){
		std::cout << "Not enought arguments\n";
		return 0;
	}
	int x = atoi(argv[1]);
	int weight = atoi(argv[2]);
	int N = atoi(argv[3]);
	std::cout << "X = " << x << " W = " << weight << " N = " << N / 1000 << "k\t";
	// GGKP test
	TestGGKP test;
	test.fullTest(N, x, weight);
*/

	if (argc < 2){
		//std::cout << "Not enought arguments\n";
		return 0;
	}
	int N = atoi(argv[1]);

	ShapeTest test;
	test.fullTest(N);
 	return 0;
}
