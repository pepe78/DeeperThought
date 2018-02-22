#include "DNNLayerRelu.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

__global__ void relu_forward(float *outp, const float *inp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		int i = tid % inputWidth;
		tid = tid / inputWidth;

		if (inp[tid * inputWidth + i] > 0)
		{
			outp[tid * inputWidth + i] = inp[tid * inputWidth + i];
		}
		else
		{
			outp[tid * inputWidth + i] = 0;
		}
	}
}

__global__ void relu_backward(float *dinp, const float *doutp, const float *outp, const float *inp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		int i = tid % inputWidth;
		tid = tid / inputWidth;

		if (inp[tid * inputWidth + i] > 0)
		{
			dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i];
		}
		else
		{
			dinp[tid * inputWidth + i] = 0;
		}
	}
}

DNNLayerRelu::DNNLayerRelu(int _inputWidth, int _batchSize)
	: DNNLayer(_batchSize, _inputWidth, _inputWidth, 0, 0, 0)
{

}

DNNLayerRelu::~DNNLayerRelu()
{

}

void DNNLayerRelu::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) * inputWidth + threadsPerBlock - 1) / threadsPerBlock;
	relu_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth) * inputWidth);
}

void DNNLayerRelu::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int threadsPerBlock = 256;
		int numBlocks = ((input->GetSize() / inputWidth) * inputWidth + threadsPerBlock - 1) / threadsPerBlock;
		relu_backward<<<numBlocks, threadsPerBlock>>> ((float*)deltaInput->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
			(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth) * inputWidth);
	}
}
