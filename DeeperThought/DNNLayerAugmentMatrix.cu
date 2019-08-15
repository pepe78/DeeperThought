#include "DNNLayerAugmentMatrix.cuh"

#include <cstdlib>
#include <cstdio>

#define MAXX1X2 784
#define MAXNUMCONVY1Y2 3920

__global__ void augmentmatrix_forward(float *outp, const float *inp, const float *pars, int numPics, int inputWidth, int outputWidth, int numConvolutions, int x1, int x2, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float pics[MAXX1X2];
		for (int i = 0; i < x1 * x2 * numPics; i++)
		{
			pics[i] = inp[tid * inputWidth + i];
		}
		float convos[MAXNUMCONVY1Y2];

		int pos = 0;
		for (int c = 0; c < numConvolutions; c++)
		{
		  for (int i = 0; i < x1 * x2; i++)
		  {
			  convos[i] = pars[c * x1 * x2 + i];
		  }

			for (int p = 0; p < numPics; p++)
			{
				for (int i = 0; i < x1; i++)
				{
					for (int j = 0; j < x2; j++)
					{
						float tmp = 0;
						for (int k = 0; k < x2; k++)
						{
							tmp += pics[p * x1 * x2 + i * x2 + k] * convos[k * x2 + j];
						}
						outp[tid * outputWidth + pos] = tmp;
						pos++;
					}
				}
			}
		}
	}
}

__global__ void augmentmatrix_backward(float *dinp, float *dpars, const float *doutp, const float *outp, const float *inp, const float *pars, int numPics, int inputWidth, int outputWidth, int numConvolutions, int x1, int x2, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float pics[MAXX1X2];
		for (int i = 0; i < x1 * x2 * numPics; i++)
		{
			pics[i] = inp[tid * inputWidth + i];
		}
		float convos[MAXNUMCONVY1Y2];

		int pos = 0;
		for (int c = 0; c < numConvolutions; c++)
		{
		  for (int i = 0; i < x1 * x2; i++)
		  {
			  convos[i] = pars[c * x1 * x2 + i];
		  }

			for (int p = 0; p < numPics; p++)
			{
				for (int i = 0; i < x1; i++)
				{
					for (int j = 0; j < x2; j++)
					{
						float tmp = doutp[tid * outputWidth + pos];
						if (tmp != 0)
						{
							for (int k = 0; k < x2; k++)
							{
								if (dinp != NULL)
								{
									dinp[tid * inputWidth + p * x1 * x2 + i * x2 + k] += tmp * convos[k * x2 + j];
								}
								atomicAdd(&(dpars[c * x1 * x2 + k * x2 + j]), tmp * pics[p * x1 * x2 + i * x2 + k]);
							}
						}
						pos++;
					}
				}
			}
		}
	}
}

DNNLayerAugmentMatrix::DNNLayerAugmentMatrix(int _numPics, int _x1, int _x2, int _numConvolutions, int _batchSize, float _initVal, float _stepSize)
	: DNNLayer(_batchSize, _numPics * _x1 * _x2, _numPics * _x1 * _x2 * _numConvolutions, _numConvolutions * _x1 * _x2, _initVal, _stepSize)
{
	x1 = _x1;
	x2 = _x2;
	if(x1 != x2)
	{
		fprintf(stderr, "supporting only squares for augment matrix layer now\n");
		exit(-1);
	}
	numPics = _numPics;
	numConvolutions = _numConvolutions;

	if (x1 * x2 * numPics > MAXX1X2)
	{
		fprintf(stderr, "Project needs to be recompiled with larger field for augment matrix layer\n");
		exit(-1);
	}
	if (x1 * x2 * numPics > MAXNUMCONVY1Y2)
	{
		fprintf(stderr, "Project needs to be recompiled with larger field for augment matrix layer\n");
		exit(-1);
	}

	//set diagonal to 1, with rest random numbers so it starts with close to identy projection
	float* pars = (float*)params->GetCPUMemory();
	for(int i=0;i<numConvolutions;i++)
	{
		for(int j=0;j<x1;j++)
		{
			pars[i * x1 * x2 + j * x2 +j] = 1.0f;
		}
	}
	params->CopyCPUtoGPU();
}

DNNLayerAugmentMatrix::~DNNLayerAugmentMatrix()
{

}

void DNNLayerAugmentMatrix::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	augmentmatrix_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), numPics, inputWidth, outputWidth, numConvolutions, x1, x2, (input->GetSize() / inputWidth));
}

void DNNLayerAugmentMatrix::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	augmentmatrix_backward<<<numBlocks, threadsPerBlock>>>(deltaInput == NULL ? NULL : (float*)deltaInput->GetGPUMemory(), (float*)dparams->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
		(float*)output->GetGPUMemory(), (float*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), numPics, inputWidth, outputWidth, numConvolutions, x1, x2, (input->GetSize() / inputWidth));
}
