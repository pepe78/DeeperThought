#include "CPUGPUMemory.cuh"
#include "RandUtils.cuh"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <climits>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

CPUGPUMemory::CPUGPUMemory(bool _is_float, size_t _size, float _initValMin, float _initValMax)
{
	is_float = _is_float;
	if (_size > INT_MAX)
	{
		fprintf(stderr, "project not ready for such sizes!");
		exit(-1);
	}
	size = (int)_size;

	memCPU = is_float ? (void*)new float[size] : (void*)new int[size];
	memset(memCPU, 0, size * (is_float ? sizeof(float) : sizeof(int)));
	if (_initValMin != 0 || _initValMax != 0)
	{
		if (is_float)
		{
			float *t = (float*)memCPU;
			for (int i = 0; i < size; i++)
			{
				t[i] = _initValMin + (_initValMax - _initValMin) * getRand();
			}
		}
	}
	cudaError_t cudaStatus = cudaMalloc((void**)&memGPU, size *  (is_float ? sizeof(float) : sizeof(int)));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(-1);
	}

	CopyCPUtoGPU();
}

void CPUGPUMemory::Resize(size_t newSize)
{
	if (newSize > INT_MAX)
	{
		fprintf(stderr, "project not ready for such sizes!");
		exit(-1);
	}
	size = (int)newSize;
}

CPUGPUMemory::~CPUGPUMemory()
{
	if (is_float)
	{
		delete[] (float*)memCPU;
	}
	else
	{
		delete[] (int*)memCPU;
	}
	cudaFree(memGPU);
}

void CPUGPUMemory::CopyCPUtoGPU()
{
	cudaError_t cudaStatus = cudaMemcpy(memGPU, memCPU, size * (is_float ? sizeof(float) : sizeof(int)), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(-1);
	}
}

void CPUGPUMemory::CopyGPUtoCPU()
{
	cudaError_t cudaStatus = cudaMemcpy(memCPU, memGPU, size * (is_float ? sizeof(float) : sizeof(int)), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(-1);
	}
}

void* CPUGPUMemory::GetCPUMemory()
{
	return memCPU;
}

void* CPUGPUMemory::GetGPUMemory()
{
	return memGPU;
}

int CPUGPUMemory::GetSize()
{
	return size;
}

void CPUGPUMemory::Reset()
{
	memset(memCPU, 0, size * (is_float ? sizeof(float) : sizeof(int)));
	CopyCPUtoGPU();
}

void CPUGPUMemory::SaveToFile(ofstream &os)
{
	CopyGPUtoCPU();
	os.write((char*)memCPU, size * (is_float ? sizeof(float) : sizeof(int)));
}

void CPUGPUMemory::LoadFromFile(ifstream &is)
{
	is.read((char*)memCPU, size * (is_float ? sizeof(float) : sizeof(int)));
	CopyCPUtoGPU();
}
