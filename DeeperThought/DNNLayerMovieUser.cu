#include "DNNLayerMovieUser.cuh"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

#define MAXVECWIDTH 1000

__global__ void movieuser_forward(float *outp, const int *inp, const float *params, int numUsers, int numMovies, int vectorWidthMovie, int vectorWidthUser, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float movie[MAXVECWIDTH];
		float user[MAXVECWIDTH];

		int m = inp[2 * tid];
		int u = inp[2 * tid + 1];

		for (int i = 0; i < vectorWidthMovie; i++)
		{
			movie[i] = params[m * vectorWidthMovie + i];
		}
		for (int i = 0; i < vectorWidthUser; i++)
		{
			user[i] = params[numMovies *vectorWidthMovie + u * vectorWidthUser + i];
		}

		float tmp = params[numMovies * vectorWidthMovie + numUsers * vectorWidthUser];
		for (int i = 0; i < vectorWidthMovie; i++)
		{
			for (int j = 0; j < vectorWidthUser; j++)
			{
				tmp += movie[i] * movie[i] * user[j] * user[j] * params[numMovies * vectorWidthMovie + numUsers * vectorWidthUser + i * vectorWidthUser + j + 1];
			}
		}
		outp[tid] = tmp;
	}
}

__global__ void movieuser_backward(float *dparams, const float *doutp, const float *outp, const int *inp, const float *params, int numUsers, int numMovies, int vectorWidthMovie, int vectorWidthUser, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		float movie[MAXVECWIDTH];
		float user[MAXVECWIDTH];
		float dmovie[MAXVECWIDTH];
		float duser[MAXVECWIDTH];

		int m = inp[2 * tid];
		int u = inp[2 * tid + 1];

		for (int i = 0; i < vectorWidthMovie; i++)
		{
			movie[i] = params[m * vectorWidthMovie + i];
			dmovie[i] = 0;
		}
		for (int i = 0; i < vectorWidthUser; i++)
		{
			user[i] = params[numMovies *vectorWidthMovie + u * vectorWidthUser + i];
			duser[i] = 0;
		}

		atomicAdd(&(dparams[numMovies * vectorWidthMovie + numUsers * vectorWidthUser]), doutp[tid]);
		for (int i = 0; i < vectorWidthMovie; i++)
		{
			for (int j = 0; j < vectorWidthUser; j++)
			{
				dmovie[i] += doutp[tid] * movie[i] * 2.0f * user[j] * user[j] * params[numMovies * vectorWidthMovie + numUsers * vectorWidthUser + i * vectorWidthUser + j + 1];
				duser[j] += doutp[tid] * movie[i] * movie[i] * user[j] * 2.0f * params[numMovies * vectorWidthMovie + numUsers * vectorWidthUser + i * vectorWidthUser + j + 1];

				atomicAdd(&(dparams[numMovies * vectorWidthMovie + numUsers * vectorWidthUser + i * vectorWidthUser + j + 1]), doutp[tid] * movie[i] * movie[i] * user[j] * user[j]);
			}
		}

		for (int i = 0; i < vectorWidthMovie; i++)
		{
			atomicAdd(&(dparams[m * vectorWidthMovie + i]), dmovie[i]);
		}
		for (int i = 0; i < vectorWidthUser; i++)
		{
			atomicAdd(&(dparams[numMovies * vectorWidthMovie + u * vectorWidthUser + i]), duser[i]);
		}
	}
}

DNNLayerMovieUser::DNNLayerMovieUser(int _numMovies, int _numUsers, int _vectorWidthMovie, int _vectorWidthUser, float _initValues, float _stepSize, int _batchSize)
	: DNNLayer(_batchSize, 2, 1, _numMovies * _vectorWidthMovie + _numUsers * _vectorWidthUser + (1 + _vectorWidthMovie * _vectorWidthUser), _initValues, _stepSize)
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
