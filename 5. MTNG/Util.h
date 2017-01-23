#ifndef __MTNG_UTIL_H__
#define __MTNG_UTIL_H__

/*class MtngInt64{
private:
	__int64 value;
public:
	MtngInt64() :value(0){
	}
	MtngInt64(const MtngInt64& val) :value(val.value){}
	MtngInt64 operator *(MtngInt64& val){
		return 
	}
};*/


typedef double MetricUnit;

#include <math.h>

#define GEOM_EPS 0.000000000001
#define SMALL_DIST 0.00000001

__device__ inline bool Equals(MetricUnit p1, MetricUnit p2){
	return fabs(p1 - p2) < GEOM_EPS;
}

//! p1>p2
/*inline bool Greater(MetricUnit p1, MetricUnit p2){
	return p1 > p2 + GEOM_EPS;
}
//! p1 >= p2
inline bool GreaterOrEquals(MetricUnit p1, MetricUnit p2){
	return p1 > p2 - GEOM_EPS;
}*/

#endif