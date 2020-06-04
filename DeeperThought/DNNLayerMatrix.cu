#include "DNNLayerMatrix.cuh"
#include "GPU.cuh"

#include <cstdlib>
#include <cstdio>
#include <chrono>

using namespace std;

#define MAXINP 5000
#define MAXOUTP 5000

__global__ void matrix_forward(float *outp, const float *inp, const float *pars, int inputWidth, int outputWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float inp2[MAXINP];
		for(int i=0;i<inputWidth;i++)
		{
			inp2[i] = inp[tid* inputWidth + i];
		}
	
		for (int i = 0; i < outputWidth; i++)
		{
			float tmp = pars[i * (inputWidth + 1)];
			for (int j = 0; j < inputWidth; j++)
			{
				tmp += pars[i * (inputWidth + 1) + 1 + j] * inp2[j];
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
		float inp2[MAXINP];
		for(int i=0;i<inputWidth;i++)
		{
			inp2[i] = inp[tid* inputWidth + i];
		}
		float doutp2[MAXOUTP];
		for(int i=0;i<outputWidth;i++)
		{
			doutp2[i] = doutp[tid * outputWidth + i];
		}
	
		for (int i = 0; i < outputWidth; i++)
		{
			atomicAdd(&(dpars[i * (inputWidth + 1)]), doutp2[i]);
			for (int j = 0; j < inputWidth; j++)
			{
				if (dinp != NULL)
				{
					dinp[tid* inputWidth + j] += doutp2[i] * pars[i * (inputWidth + 1) + 1 + j];
				}
				atomicAdd(&(dpars[i * (inputWidth + 1) + 1 + j]), doutp2[i] * inp2[j]);
			}
		}
	}
}

DNNLayerMatrix::DNNLayerMatrix(int _inputWidth, int _outputWidth, int _batchSize, float _initVal, float _stepSize)
	: DNNLayer(_batchSize, _inputWidth, _outputWidth, _outputWidth * (_inputWidth + 1), _initVal, _stepSize)
{
	if (_inputWidth > MAXINP || _outputWidth > MAXOUTP)
	{
		fprintf(stderr, "Project needs to be recompiled with larger field for matrix layer\n");
		exit(-1);
	}
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
	matrix_backward<<<numBlocks, threadsPerBlock>>>(deltaInput == NULL ? NULL : (float*)deltaInput->GetGPUMemory(), (float*)dparams->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), inputWidth, outputWidth, (input->GetSize() / inputWidth));
}
