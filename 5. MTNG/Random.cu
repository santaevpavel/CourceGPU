#include "random.h"

uint32_t Random::x = 123456789;
uint32_t Random::y = 362436069;
uint32_t Random::z = 521288629;

Random::RandomAlgorithm Random::rndAlg = RandomAlgorithm::randXorshf96;