#include "DNNLayerError.cuh"

#include "GPU.cuh"

#include <cstdlib>

__global__ void error_square_kernel(float *error, float *dinput, const float *expOutp, const float *outp, int inputWidth, int batchSize)
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

__global__ void error_log_kernel(float *error, float *dinput, const float *expOutp, const float *outp, int inputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			if (expOutp[tid * inputWidth + i] > 0.5f)
			{
				error[tid * inputWidth + i] = -log(0.001f + outp[tid * inputWidth + i]);
				dinput[tid * inputWidth + i] = -1.0f / (0.001f + outp[tid * inputWidth + i]);
			}
			else
			{
				error[tid * inputWidth + i] = -log(1.001f - outp[tid * inputWidth + i]);
				dinput[tid * inputWidth + i] = 1.0f / (1.001f - outp[tid * inputWidth + i]);
			}
		}
	}
}

DNNLayerError::DNNLayerError(int _inputWidth, int _batchSize, bool _square)
{
	square = _square;
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
	if (square)
	{
		error_square_kernel<<<numBlocks, threadsPerBlock>>> (
			(float*)error->GetGPUMemory(), (float*)deltaInput->GetGPUMemory(), (float*)expectedOutput->GetGPUMemory(), (float*)output->GetGPUMemory(), inputWidth, (output->GetSize() / inputWidth));
		WaitForGPUToFinish();
	}
	else
	{
		error_log_kernel<<<numBlocks, threadsPerBlock>>> (
			(float*)error->GetGPUMemory(), (float*)deltaInput->GetGPUMemory(), (float*)expectedOutput->GetGPUMemory(), (float*)output->GetGPUMemory(), inputWidth, (output->GetSize() / inputWidth));
		WaitForGPUToFinish();
	}

	error->CopyGPUtoCPU();
	double ret = 0.0;
	float *m = (float*)error->GetCPUMemory();
	for (int i = 0; i < expectedOutput->GetSize(); i++)
	{
		ret += m[i];
	}

	return ret;
}

CPUGPUMemory* DNNLayerError::GetDeltaInput()
{
	return deltaInput;
}