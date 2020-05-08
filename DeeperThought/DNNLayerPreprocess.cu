#include "DNNLayerPreprocess.cuh"
#include "RandUtils.cuh"

#include <cstdlib>
#include <cstdio>

__global__ void preprocess_forward(float *outp, float *tmp, const float *noise, const float *inp, int inputWidth, int outputWidth, int x1, int x2, int batchSize, float angle, float sx, float sy, int nx1, int nx2, bool trainRun)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		if(trainRun)
		{
			for(int i=0;i<outputWidth;i++)
			{
				outp[tid * outputWidth + i] = 0.0f;
				tmp[tid * outputWidth + i] = 0.0f;
			}
		
			for(int i=0;i<nx1;i++)
			{
				for(int j=0;j<nx2;j++)
				{
					int ii = (int)i * x1 / nx1;
					int jj = (int)j * x2 / nx2;
					float p = inp[tid * inputWidth + ii * x2 + jj];
					
					float iii = (i + 0.0f) / (nx1 + 0.0f) - 0.5f;
					float jjj = (j + 0.0f) / (nx2 + 0.0f) - 0.5f;
					
					float iiii = (cos(angle) * iii - sin(angle) * jjj) * sx + 0.5f;
					float jjjj = (sin(angle) * iii + cos(angle) * jjj) * sy + 0.5f;
					
					int fi = (int)((float)iiii * (x1 + 0.0f));
					int fj = (int)((float)jjjj * (x2 + 0.0f));
					
					if(fi>=0 && fi<x1 && fj>=0 && fj<x2)
					{
						outp[tid * outputWidth + fi * x2 + fj] = (outp[tid * outputWidth + fi * x2 + fj] * tmp[tid * outputWidth + fi * x2 + fj] + p) / (tmp[tid * outputWidth + fi * x2 + fj] + 1.0f);
						tmp[tid * outputWidth + fi * x2 + fj] += 1.0f;
					}
				}
			}
			for(int i=0;i<outputWidth;i++)
			{
				outp[tid * outputWidth + i] += noise[tid * outputWidth + i];
			}	
		}
		else
		{
			for(int i=0;i<outputWidth;i++)
			{
				outp[tid * outputWidth + i] = inp[tid * outputWidth + i];
			}	
		}
	}
}

DNNLayerPreprocess::DNNLayerPreprocess(int _x1, int _x2, int _batchSize, float _minAngle, float _maxAngle, float _minStretch, float _maxStretch, float _minNoise, float _maxNoise, int _x1SamplePoints, int _x2SamplePoints)
	: DNNLayer(_batchSize, _x1 * _x2, _x1 * _x2, 0, 0, 0)
{
	x1 = _x1;
	x2 = _x2;
	dom = new CPUGPUMemory(true, _batchSize * _x1 * _x2, 0);
	tmpMem = new CPUGPUMemory(true, _batchSize * _x1 * _x2, 0);
	
	minAngle = _minAngle;
	maxAngle = _maxAngle;
	minStretch = _minStretch;
	maxStretch = _maxStretch;
	minNoise = _minNoise;
	maxNoise = _maxNoise;
	x1SamplePoints = _x1SamplePoints;
	x2SamplePoints = _x2SamplePoints;
}

DNNLayerPreprocess::~DNNLayerPreprocess()
{
	delete dom;
	delete tmpMem;
}

void DNNLayerPreprocess::Forward(CPUGPUMemory* input)
{
	if (trainRun)
	{
		float *p = (float*)dom->GetCPUMemory();
		for (int i = 0; i < dom->GetSize(); i++)
		{
			p[i] = minNoise + getRand() * (maxNoise - minNoise);
		}
		dom->CopyCPUtoGPU();
	}
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	preprocess_forward<<<numBlocks, threadsPerBlock>>>(
		(float*)output->GetGPUMemory(), (float*)tmpMem->GetGPUMemory(), (float*)dom->GetGPUMemory(), (float*)input->GetGPUMemory(), inputWidth, outputWidth, x1, x2, (input->GetSize() / inputWidth),
		minAngle + getRand() * (maxAngle - minAngle), minStretch + getRand() * (maxStretch - minStretch), minStretch + getRand() * (maxStretch - minStretch), 
		x1SamplePoints, x2SamplePoints, trainRun);

}

void DNNLayerPreprocess::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
}
