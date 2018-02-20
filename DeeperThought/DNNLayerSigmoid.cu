#include "DNNLayerSigmoid.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

__global__ void sigmoid_forward(float *outp, const float *inp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			outp[tid * inputWidth + i] = 1.0f / (1.0f + exp(inp[tid * inputWidth + i]));
		}
	}
}

__global__ void sigmoid_backward(float *dinp, const float *doutp, const float *outp, const float *inp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i] * outp[tid * inputWidth + i] * (outp[tid * inputWidth + i] - 1.0f);
		}
	}
}

DNNLayerSigmoid::DNNLayerSigmoid(int _inputWidth, int _batchSize)
	: DNNLayer(_batchSize, _inputWidth, _inputWidth, 0, 0, 0)
{

}

DNNLayerSigmoid::~DNNLayerSigmoid()
{

}

void DNNLayerSigmoid::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	sigmoid_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth));
}

void DNNLayerSigmoid::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int threadsPerBlock = 256;
		int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
		sigmoid_backward << <numBlocks, threadsPerBlock >> > ((float*)deltaInput->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
			(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth));
	}
}
