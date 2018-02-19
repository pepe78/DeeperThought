#ifndef GPUCUH
#define GPUCUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

void InitGPU();
void WaitForGPUToFinish();
void ReleaseGPU();

#endif
