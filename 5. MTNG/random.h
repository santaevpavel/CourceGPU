#ifndef __MTNG_RANDOM_H__
#define __MTNG_RANDOM_H__

#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdlib>


class Random{
	public:
		enum RandomAlgorithm{ randXorshf96, randStd};
	private:
		static uint32_t x;
		static uint32_t y;
		static uint32_t z;

		inline static uint32_t xorshf96() {          //period 2^96-1
			uint32_t t;
			x ^= x << 16;
			x ^= x >> 5;
			x ^= x << 1;

			t = x;
			x = y;
			y = z;
			z = t ^ x ^ y;

			return z;
		}
	public:

		static void Init(){
		}

		static RandomAlgorithm rndAlg;
		//! Функция возвражает случайное число от 
		static double rand01(){
			uint32_t code = 0;
			int d = 0;
			switch (rndAlg){
				case randXorshf96:
					code = xorshf96();
					return code / (double)std::numeric_limits<uint32_t>::max();
				case randStd:
					d = rand();
					return (double)(d) / (RAND_MAX);
				default:
					return rand();
			}
		}
};

#endif