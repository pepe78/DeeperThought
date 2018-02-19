#include "DNNLayerMax.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

__global__ void max_forward(float *outp, const float *inp, int inputWidth, int outputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < outputWidth; i++)
		{
			float tmp = -FLT_MAX;
			for (int j = 0; j < inputWidth / outputWidth; j++)
			{
				if (tmp < inp[tid * inputWidth + i * (inputWidth / outputWidth) + j])
				{
					tmp = inp[tid * inputWidth + i * (inputWidth / outputWidth) + j];
				}
			}
			outp[tid * outputWidth + i] = tmp;
		}
	}
}

__global__ void max_backward(float *dinp, const float *doutp, const float *outp, const float *inp, int inputWidth, int outputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < outputWidth; i++)
		{
			float tmp = -FLT_MAX;
			int pos = -1;
			for (int j = 0; j < inputWidth / outputWidth; j++)
			{
				if (tmp < inp[tid * inputWidth + i * (inputWidth / outputWidth) + j])
				{
					tmp = inp[tid * inputWidth + i * (inputWidth / outputWidth) + j];
					pos = j;
				}
			}
			dinp[tid * inputWidth + i * (inputWidth / outputWidth) + pos] += doutp[tid * outputWidth + i];
		}
	}
}

DNNLayerMax::DNNLayerMax(int _inputWidth, int _outputWidth, int _batchSize)
	: DNNLayer(_batchSize, _inputWidth, _outputWidth, 0, 0, 0)
{
	if (inputWidth % outputWidth != 0)
	{
		fprintf(stderr, "max layer parameters wrong!\n");
		exit(-1);
	}
}

DNNLayerMax::~DNNLayerMax()
{

}

void DNNLayerMax::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	max_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, outputWidth, (input->GetSize() / inputWidth));
}

void DNNLayerMax::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	max_backward<<<numBlocks, threadsPerBlock>>>((float*)deltaInput->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, outputWidth, (input->GetSize() / inputWidth));
}
