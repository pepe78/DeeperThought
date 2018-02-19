#include "RandUtils.cuh"

#include <cstdlib>
#include <ctime>

void InitRand()
{
	srand((unsigned int)time(NULL));
}

float getRand()
{
	return (float)(rand() + 0.0f) / (0.0f + RAND_MAX);
}
