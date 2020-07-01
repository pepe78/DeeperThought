#include "DNNLayerLn.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

__global__ void ln_forward(float *outp, const float *inp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			outp[tid * inputWidth + i] = log(inp[tid * inputWidth + i]+0.01f);
		}
	}
}

__global__ void ln_backward(float *dinp, const float *doutp, const float *outp, const float *inp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i] / (inp[tid * inputWidth + i] + 0.01f);
		}
	}
}

DNNLayerLn::DNNLayerLn(int _inputWidth, int _batchSize)
	: DNNLayer(_batchSize, _inputWidth, _inputWidth, 0, 0, 0)
{
}

DNNLayerLn::~DNNLayerLn()
{

}

void DNNLayerLn::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	ln_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth));
}

void DNNLayerLn::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int threadsPerBlock = 256;
		int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
		ln_backward << <numBlocks, threadsPerBlock >> > ((float*)deltaInput->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
			(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, (input->GetSize() / inputWidth));
	}
}
