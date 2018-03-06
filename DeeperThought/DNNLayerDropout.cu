#include "DNNLayerDropout.cuh"
#include "RandUtils.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

__global__ void dropout_forward(float *outp, const float *inp, const float *dom, int inputWidth, float perc, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		int i = tid % inputWidth;
		tid = tid / inputWidth;

		if (dom[tid * inputWidth + i] < 1.0f - perc)
		{
			outp[tid * inputWidth + i] = inp[tid * inputWidth + i];
		}
		else
		{
			outp[tid * inputWidth + i] = 0;
		}
	}
}

__global__ void dropout_backward(float *dinp, const float *doutp, const float *outp, const float *inp, const float *dom, int inputWidth, float perc, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		int i = tid % inputWidth;
		tid = tid / inputWidth;

		if (dom[tid * inputWidth + i] < 1.0f - perc)
		{
			dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i];
		}
	}
}

DNNLayerDropout::DNNLayerDropout(int _inputWidth, int _batchSize, float _perc)
	: DNNLayer(_batchSize, _inputWidth, _inputWidth, 0, 0, 0)
{
	dom = new CPUGPUMemory(true, _batchSize * _inputWidth, 0);
	perc = _perc;
}

DNNLayerDropout::~DNNLayerDropout()
{
	delete dom;
}

void DNNLayerDropout::Forward(CPUGPUMemory* input)
{
	if (trainRun)
	{
		float *p = (float*)dom->GetCPUMemory();
		for (int i = 0; i < dom->GetSize(); i++)
		{
			p[i] = getRand();
		}
		dom->CopyCPUtoGPU();
	}
	else
	{
		dom->Reset();
	}

	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) * inputWidth + threadsPerBlock - 1) / threadsPerBlock;
	dropout_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), (float*)dom->GetGPUMemory(), inputWidth, perc, (input->GetSize() / inputWidth) * inputWidth);
}

void DNNLayerDropout::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int threadsPerBlock = 256;
		int numBlocks = ((input->GetSize() / inputWidth) * inputWidth + threadsPerBlock - 1) / threadsPerBlock;
		dropout_backward << <numBlocks, threadsPerBlock >> > ((float*)deltaInput->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
			(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), (float*)dom->GetGPUMemory(), inputWidth, perc, (input->GetSize() / inputWidth) * inputWidth);
	}
}
