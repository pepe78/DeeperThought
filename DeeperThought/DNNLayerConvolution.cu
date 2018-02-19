#include "DNNLayerConvolution.cuh"

#include <cstdlib>
#include <cstdio>

#define MAXX1X2 784
#define MAXNUMCONVY1Y2 8100

__global__ void convolution_forward(float *outp, const float *inp, const float *pars, int inputWidth, int outputWidth, int numConvolutions, int x1, int x2, int y1, int y2, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float x[MAXX1X2];
		for (int i = 0; i < x1 * x2; i++)
		{
			x[i] = inp[tid * inputWidth + i];
		}
		float p[MAXNUMCONVY1Y2];
		for (int i = 0; i < numConvolutions * y1 * y2; i++)
		{
			p[i] = pars[i];
		}

		int pos = 0;
		for (int c = 0; c < numConvolutions; c++)
		{
			for (int i = 0; i < x1 - y1 + 1; i++)
			{
				for (int j = 0; j < x2 - y2 + 1; j++)
				{
					float tmp = 0;
					for (int k = 0; k < y1; k++)
					{
						for (int l = 0; l < y2; l++)
						{
							tmp += x[(i + k) * x2 + (j + l)] * p[c * y1 * y2 + k * y2 + l];
						}
					}
					outp[tid * outputWidth + pos] = tmp;
					pos++;
				}
			}
		}
	}
}

__global__ void convolution_backward(float *dinp, float *dpars, const float *doutp, const float *outp, const float *inp, const float *pars, int inputWidth, int outputWidth, int numConvolutions, int x1, int x2, int y1, int y2, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float x[MAXX1X2];
		for (int i = 0; i < x1 * x2; i++)
		{
			x[i] = inp[tid * inputWidth + i];
		}
		float p[MAXNUMCONVY1Y2];
		for (int i = 0; i < numConvolutions * y1 * y2; i++)
		{
			p[i] = pars[i];
		}

		int pos = 0;
		for (int c = 0; c < numConvolutions; c++)
		{
			for (int i = 0; i < x1 - y1 + 1; i++)
			{
				for (int j = 0; j < x2 - y2 + 1; j++)
				{
					float tmp = doutp[tid * outputWidth + pos];
					if (tmp != 0)
					{
						for (int k = 0; k < y1; k++)
						{
							for (int l = 0; l < y2; l++)
							{
								if (dinp != NULL)
								{
									dinp[tid * inputWidth + (i + k) * x2 + (j + l)] += tmp * p[c * y1 * y2 + k * y2 + l];
								}
								atomicAdd(&(dpars[c * y1 * y2 + k * y2 + l]), tmp * x[(i + k) * x2 + (j + l)]);
							}
						}
					}
					pos++;
				}
			}
		}
	}
}

DNNLayerConvolution::DNNLayerConvolution(int _x1, int _x2, int _outputWidth, int _numConvolutions, int _y1, int _y2, int _batchSize, float _initVal, float _stepSize)
	: DNNLayer(_batchSize, _x1 * _x2, _outputWidth, _numConvolutions * _y1 * _y2, _initVal, _stepSize)
{
	x1 = _x1;
	x2 = _x2;
	y1 = _y1;
	y2 = _y2;
	numConvolutions = _numConvolutions;

	if ((x1 - y1 + 1) * (x2 - y2 + 1) * numConvolutions != outputWidth)
	{
		fprintf(stderr, "convolution layer parameters wrong!\n");
		exit(-1);
	}

	if (x1 * x2 > MAXX1X2)
	{
		fprintf(stderr, "Project needs to be recompiled with larger field for convolution layer\n");
		exit(-1);
	}
	if (numConvolutions * y1 * y2 > MAXNUMCONVY1Y2)
	{
		fprintf(stderr, "Project needs to be recompiled with larger field for convolution layer\n");
		exit(-1);
	}
}

DNNLayerConvolution::~DNNLayerConvolution()
{

}

void DNNLayerConvolution::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	convolution_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), inputWidth, outputWidth, numConvolutions, x1, x2, y1, y2, (input->GetSize() / inputWidth));
}

void DNNLayerConvolution::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	convolution_backward<<<numBlocks, threadsPerBlock>>>(deltaInput == NULL ? NULL : (float*)deltaInput->GetGPUMemory(), (float*)dparams->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), inputWidth, outputWidth, numConvolutions, x1, x2, y1, y2, (input->GetSize() / inputWidth));
}
