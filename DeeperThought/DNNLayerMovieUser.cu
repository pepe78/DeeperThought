#include "DNNLayerMovieUser.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

#define MAXVECWIDTH 100

__global__ void movieuser_forward(float *outp, const int *inp, const float *params, int numUsers, int numMovies, int vectorWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float movie[MAXVECWIDTH];
		float user[MAXVECWIDTH];

		int m = inp[2 * tid];
		int u = inp[2 * tid];

		for (int i = 0; i < vectorWidth; i++)
		{
			movie[i] = params[m * vectorWidth + i];
			user[i] = params[(numMovies + u) * vectorWidth + i];
		}

		for (int i = 0; i < vectorWidth; i++)
		{
			for (int j = 0; j < vectorWidth; j++)
			{
				outp[tid * vectorWidth * vectorWidth + i * vectorWidth + j] = movie[i] * user[j];
			}
		}
	}
}

__global__ void movieuser_backward(float *dparams, const float *doutp, const float *outp, const int *inp, const float *params, int numUsers, int numMovies, int vectorWidth, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float movie[MAXVECWIDTH];
		float user[MAXVECWIDTH];
		float dmovie[MAXVECWIDTH];
		float duser[MAXVECWIDTH];

		int m = inp[2 * tid];
		int u = inp[2 * tid];

		for (int i = 0; i < vectorWidth; i++)
		{
			movie[i] = params[m * vectorWidth + i];
			user[i] = params[(numMovies + u) * vectorWidth + i];
			dmovie[i] = 0;
			duser[i] = 0;
		}

		for (int i = 0; i < vectorWidth; i++)
		{
			for (int j = 0; j < vectorWidth; j++)
			{
				dmovie[i] += doutp[tid * vectorWidth * vectorWidth + i * vectorWidth + j] * user[j];
				duser[j] += doutp[tid * vectorWidth * vectorWidth + i * vectorWidth + j] * movie[i];
			}
		}

		for (int i = 0; i < vectorWidth; i++)
		{
			atomicAdd(&(dparams[m * vectorWidth + i]), dmovie[i]);
			atomicAdd(&(dparams[(numMovies + u) * vectorWidth + i]), duser[i]);
		}
	}
}

DNNLayerMovieUser::DNNLayerMovieUser(int _numMovies, int _numUsers, int _vectorWidth, float _initValues, float _stepSize, int _batchSize)
	: DNNLayer(_batchSize, 2, _vectorWidth * _vectorWidth, (_numMovies + _numUsers) * _vectorWidth, _initValues, _stepSize)
{
	numMovies = _numMovies;
	numUsers = _numUsers;
	vectorWidth = _vectorWidth;
}

DNNLayerMovieUser::~DNNLayerMovieUser()
{

}

void DNNLayerMovieUser::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	movieuser_forward << <numBlocks, threadsPerBlock >> >(
		(float*)output->GetGPUMemory(), (int*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), numUsers, numMovies, vectorWidth, (input->GetSize() / inputWidth));
}

void DNNLayerMovieUser::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	movieuser_backward << <numBlocks, threadsPerBlock >> > ((float*)dparams->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
		(float*)output->GetGPUMemory(), (int*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), numUsers, numMovies, vectorWidth, (input->GetSize() / inputWidth));
}
