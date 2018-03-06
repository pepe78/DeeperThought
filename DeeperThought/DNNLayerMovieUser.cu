#include "DNNLayerMovieUser.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

__global__ void movieuser_forward(float *outp, const int *inp, const float *params, int numUsers, int numMovies, int vectorWidthMovie, int vectorWidthUser, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		int m = inp[2 * tid];
		int u = inp[2 * tid + 1];

		for (int i = 0; i < vectorWidthMovie; i++)
		{
			outp[tid * (vectorWidthMovie + vectorWidthUser) + i] = params[m * vectorWidthMovie + i];
		}
		for (int i = 0; i < vectorWidthUser; i++)
		{
			outp[tid * (vectorWidthMovie + vectorWidthUser) + vectorWidthMovie + i] = params[numMovies * vectorWidthMovie + u * vectorWidthUser + i];
		}
	}
}

__global__ void movieuser_backward(float *dparams, const float *doutp, const float *outp, const int *inp, const float *params, int numUsers, int numMovies, int vectorWidthMovie, int vectorWidthUser, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		int m = inp[2 * tid];
		int u = inp[2 * tid + 1];

		for (int i = 0; i < vectorWidthMovie; i++)
		{
			atomicAdd(&(dparams[m * vectorWidthMovie + i]), doutp[tid * (vectorWidthMovie + vectorWidthUser) + i]);
		}
		for (int i = 0; i < vectorWidthUser; i++)
		{
			atomicAdd(&(dparams[numMovies * vectorWidthMovie + u * vectorWidthUser + i]), doutp[tid * (vectorWidthMovie + vectorWidthUser) + vectorWidthMovie + i]);
		}
	}
}

DNNLayerMovieUser::DNNLayerMovieUser(int _numMovies, int _numUsers, int _vectorWidthMovie, int _vectorWidthUser, float _initValues, float _stepSize, int _batchSize)
	: DNNLayer(_batchSize, 2, _vectorWidthMovie + _vectorWidthUser, _numMovies * _vectorWidthMovie + _numUsers * _vectorWidthUser, _initValues, _stepSize)
{
	numMovies = _numMovies;
	numUsers = _numUsers;
	vectorWidthMovie = _vectorWidthMovie;
	vectorWidthUser = _vectorWidthUser;
}

DNNLayerMovieUser::~DNNLayerMovieUser()
{

}

void DNNLayerMovieUser::Forward(CPUGPUMemory* input)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	movieuser_forward << <numBlocks, threadsPerBlock >> >(
		(float*)output->GetGPUMemory(), (int*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), numUsers, numMovies, vectorWidthMovie, vectorWidthUser, (input->GetSize() / inputWidth));
}

void DNNLayerMovieUser::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	movieuser_backward << <numBlocks, threadsPerBlock >> > ((float*)dparams->GetGPUMemory(), (float*)deltaOutput->GetGPUMemory(),
		(float*)output->GetGPUMemory(), (int*)input->GetGPUMemory(), (float*)params->GetGPUMemory(), numUsers, numMovies, vectorWidthMovie, vectorWidthUser, (input->GetSize() / inputWidth));
}
