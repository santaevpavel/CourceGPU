#ifndef __MTNG_UTIL_H__
#define __MTNG_UTIL_H__

typedef double MetricUnit;

#include <math.h>

#define GEOM_EPS 0.000000000001
#define SMALL_DIST 0.00000001

__device__ inline bool Equals(MetricUnit p1, MetricUnit p2){
	return fabs(p1 - p2) < GEOM_EPS;
}

#endif