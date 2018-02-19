#include "DNNLayerMatrix.cuh"

#include <cstdlib>
#include <cstdio>

__global__ void matrix_forward(float *outp, const float *inp, const float *pars, int inputWidth, int outputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < outputWidth; i++)
		{
			float tmp = pars[i * (inputWidth + 1)];
			for (int j = 0; j < inputWidth; j++)
			{
				tmp += pars[i * (inputWidth + 1) + 1 + j] * inp[tid* inputWidth + j];
			}
			outp[tid * outputWidth + i] = tmp;
		}
	}
}

__global__ void matrix_backward(float *dinp, float *dpars, const float *doutp, const float *outp, const float *inp, const float *pars, int inputWidth, int outputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		for (int i = 0; i < outputWidth; i++)
		{
			atomicAdd(&(dpars[i * (inputWidth + 1)]), doutp[tid * outputWidth + i]);
			for (int j = 0; j < inputWidth; j++)
			{
				dinp[tid* inputWidth + j] += doutp[tid * outputWidth + i] * pars[i * (inputWidth + 1) + 1 + j];
				atomicAdd(&(dpars[i * (inputWidth + 1) + 1 + j]), doutp[tid * outputWidth + i] * inp[tid* inputWidth + j]);
			}
		}
	}
}

DNNLayerMatrix::DNNLayerMatrix(int _inputWidth, int _outputWidth, int _batchSize, float _initVal, float _stepSize)
	: DNNLayer(_batchSize, _inputWidth, _outputWidth, _outputWidth * (_inputWidth + 1), _initVal, _stepSize)
{
}

DNNLayerMatrix::~DNNLayerMatrix()
{

}

void DNNLayerMatrix::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	matrix_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), inputWidth, outputWidth, (input->GetSize() / inputWidth));
}

void DNNLayerMatrix::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	matrix_backward<<<numBlocks, threadsPerBlock>>>((float*)deltaInput->GetGPUMemory(), (float*)dparams->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), inputWidth, outputWidth, (input->GetSize() / inputWidth));
}
