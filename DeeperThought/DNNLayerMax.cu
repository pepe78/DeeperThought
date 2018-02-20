#include "DNNLayerMax.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

__global__ void max_forward(float *outp, const float *inp, int numPics, int x1, int x2, int d1, int d2, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int p = 0; p < numPics; p++)
		{
			for (int i1 = 0; i1 < x1/d1; i1++)
			{
				for (int i2 = 0; i2 < x2 / d2; i2++)
				{
					float tmp = -FLT_MAX;
					for (int j1 = 0; j1 < d1; j1++)
					{
						for (int j2 = 0; j2 < d2; j2++)
						{
							if (tmp < inp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + j1) * x2 + (i2 * d2 + j2)])
							{
								tmp = inp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + j1) * x2 + (i2 * d2 + j2)];
							}
						}
					}
					outp[tid * numPics * (x1 / d1) * (x2 / d2) + p * (x1 / d1) * (x2 / d2) + i1 * (x2 / d2) + i2] = tmp;
				}
			}
		}
	}
}

__global__ void max_backward(float *dinp, const float *doutp, const float *outp, const float *inp, int numPics, int x1, int x2, int d1, int d2, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int p = 0; p < numPics; p++)
		{
			for (int i1 = 0; i1 < x1 / d1; i1++)
			{
				for (int i2 = 0; i2 < x2 / d2; i2++)
				{
					float tmp = -FLT_MAX;
					int pos1 = -1;
					int pos2 = -1;
					for (int j1 = 0; j1 < d1; j1++)
					{
						for (int j2 = 0; j2 < d2; j2++)
						{
							if (tmp < inp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + j1) * x2 + (i2 * d2 + j2)])
							{
								tmp = inp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + j1) * x2 + (i2 * d2 + j2)];
								pos1 = j1;
								pos2 = j2;
							}
						}
					}
					dinp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + pos1) * x2 + (i2 * d2 + pos2)] += doutp[tid * numPics * (x1 / d1) * (x2 / d2) + p * (x1 / d1) * (x2 / d2) + i1 * (x2 / d2) + i2];
				}
			}
		}
	}
}

DNNLayerMax::DNNLayerMax(int _numPics, int _x1, int _x2, int _d1, int _d2, int _batchSize)
	: DNNLayer(_batchSize, _numPics * _x1 * _x2, _numPics * (_x1 / _d1) * (_x2 / _d2), 0, 0, 0)
{
	numPics = _numPics;
	x1 = _x1;
	x2 = _x2;
	d1 = _d1;
	d2 = _d2;
}

DNNLayerMax::~DNNLayerMax()
{

}

void DNNLayerMax::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	max_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), numPics, x1, x2, d1, d2, (input->GetSize() / inputWidth));
}

void DNNLayerMax::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int threadsPerBlock = 256;
		int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
		max_backward << <numBlocks, threadsPerBlock >> > ((float*)deltaInput->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
			(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), numPics, x1, x2, d1, d2, (input->GetSize() / inputWidth));
	}
}
