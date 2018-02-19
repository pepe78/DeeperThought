#include "DNNLayerError.cuh"

#include "GPU.cuh"

#include <cstdlib>

__global__ void error_kernel(float *error, float *dinput, const float *expOutp, const float *outp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			float tmp = outp[tid * inputWidth + i] - expOutp[tid * inputWidth + i];
			error[tid * inputWidth + i] = tmp * tmp;
			dinput[tid * inputWidth + i] = 2.0f * tmp;
		}
	}
}

DNNLayerError::DNNLayerError(int _inputWidth, int _batchSize)
{
	inputWidth = _inputWidth;
	batchSize = _batchSize;
	deltaInput = new CPUGPUMemory(true, inputWidth * batchSize, 0);
	error = new CPUGPUMemory(true, inputWidth * batchSize, 0);
}

DNNLayerError::~DNNLayerError()
{
	delete deltaInput;
	delete error;
}

double DNNLayerError::ComputeError(CPUGPUMemory* output, CPUGPUMemory *expectedOutput)
{
	int threadsPerBlock = 256;
	int numBlocks = ((output->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	error_kernel<<<numBlocks, threadsPerBlock>>> (
		(float*)error->GetGPUMemory(), (float*)deltaInput->GetGPUMemory(), (float*)expectedOutput->GetGPUMemory(), (float*)output->GetGPUMemory(), inputWidth, (output->GetSize() / inputWidth));
	WaitForGPUToFinish();

	error->CopyGPUtoCPU();
	double ret = 0.0;
	float *m = (float*)error->GetCPUMemory();
	for (int i = 0; i < inputWidth * batchSize; i++)
	{
		ret += m[i];
	}

	return ret;
}

CPUGPUMemory* DNNLayerError::GetDeltaInput()
{
	return deltaInput;
}