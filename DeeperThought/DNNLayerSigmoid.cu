#include "DNNLayerSigmoid.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

__global__ void sigmoid_forward(float *outp, const float *inp, int inputWidth, int batchSize, float o_min, float o_max)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			outp[tid * inputWidth + i] = o_min + (o_max - o_min) / (1.0f + exp(inp[tid * inputWidth + i]));
		}
	}
}

__global__ void sigmoid_backward(float *dinp, const float *doutp, const float *outp, const float *inp, int inputWidth, int batchSize, float o_min, float o_max)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			float tmp = (outp[tid * inputWidth + i] - o_min) / (o_max - o_min);
			dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i] * tmp * (tmp - 1.0f) * (o_max - o_min);
		}
	}
}

DNNLayerSigmoid::DNNLayerSigmoid(int _inputWidth, float _min, float _max, int _batchSize)
	: DNNLayer(_batchSize, _inputWidth, _inputWidth, 0, 0, 0)
{
	o_min = _min;
	o_max = _max;
}

DNNLayerSigmoid::~DNNLayerSigmoid()
{

}

void DNNLayerSigmoid::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	sigmoid_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth), o_min, o_max);
}

void DNNLayerSigmoid::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int threadsPerBlock = 256;
		int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
		sigmoid_backward << <numBlocks, threadsPerBlock >> > ((float*)deltaInput->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
			(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth), o_min, o_max);
	}
}
