#ifndef _COLOR_H_
#define _COLOR_H_

class Color{
public:
	int r;
	int g;
	int b;
public:
	__device__ __host__ Color() :r(0), g(0), b(0){}

	__device__ __host__ Color(int r, int g, int b) : r(r), g(g), b(b){}

};
#endif