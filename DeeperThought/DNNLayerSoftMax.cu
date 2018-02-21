#include "DNNLayerSoftMax.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

__global__ void softmax_forward(float *outp, const float *inp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float sum = 0.0f;
		for (int i = 0; i < inputWidth; i++)
		{
			float tmp = exp(inp[tid * inputWidth + i]);
			outp[tid * inputWidth + i] = tmp;
			sum += tmp;
		}
		for (int i = 0; i < inputWidth; i++)
		{
			outp[tid * inputWidth + i] /= sum;
		}
	}
}

__global__ void softmax_backward(float *dinp, const float *doutp, const float *outp, const float *inp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float sum = 0.0f;
		for (int i = 0; i < inputWidth; i++)
		{
			float tmp = exp(inp[tid * inputWidth + i]);
			sum += tmp;
		}
		for (int i = 0; i < inputWidth; i++)
		{
			dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i] * (sum - exp(inp[tid * inputWidth + i])) / (sum * sum) * exp(inp[tid * inputWidth + i]);
		}
	}
}

DNNLayerSoftMax::DNNLayerSoftMax(int _inputWidth, int _batchSize)
	: DNNLayer(_batchSize, _inputWidth, _inputWidth, 0, 0, 0)
{

}

DNNLayerSoftMax::~DNNLayerSoftMax()
{

}

void DNNLayerSoftMax::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	softmax_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth));
}

void DNNLayerSoftMax::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int threadsPerBlock = 256;
		int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
		softmax_backward<<<numBlocks, threadsPerBlock>>>((float*)deltaInput->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
			(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth));
	}
}
