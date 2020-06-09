#include "DNNLayer.cuh"

#include "GPU.cuh"

#include <cstdlib>
#include <cstdio>

__global__ void make_step_kernel(float *pars, float *dpars, float stepSize, int batchSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < batchSize)
	{
		pars[tid] -= dpars[tid] * stepSize;
		dpars[tid] = 0;
	}
}

DNNLayer::DNNLayer(int _batchSize, int _inputWidth, int _outputWidth, int _numParams, float _initVal, float _stepSize)
{
	inputWidth = _inputWidth;
	outputWidth = _outputWidth;
	numParams = _numParams;

	stepSize = _stepSize;

	deltaInput = new CPUGPUMemory(true, inputWidth * _batchSize, 0);
	output = new CPUGPUMemory(true, outputWidth * _batchSize, 0);

	if (numParams == 0)
	{
		params = NULL;
		dparams = NULL;
	}
	else
	{
		params = new CPUGPUMemory(true, numParams, _initVal);
		dparams = new CPUGPUMemory(true, numParams, 0);
	}
}

DNNLayer::~DNNLayer()
{
	if (deltaInput != NULL)
	{
		delete deltaInput;
	}
	if (params != NULL)
	{
		delete params;
	}
	if (dparams != NULL)
	{
		delete dparams;
	}
	if (output != NULL)
	{
		delete output;
	}
}

int DNNLayer::GetNumParams()
{
	return numParams;
}

void DNNLayer::Forward(CPUGPUMemory* input)
{
	fprintf(stderr, "forward not implemented!\n");
	exit(-1);
}

void DNNLayer::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	fprintf(stderr, "backward not implemented!\n");
	exit(-1);
}

void DNNLayer::ResetDeltaInput()
{
	if (deltaInput != NULL)
	{
		deltaInput->Reset();
	}
}

int DNNLayer::GetInputWidth()
{
	return inputWidth;
}

int DNNLayer::GetOutputWidth()
{
	return outputWidth;
}

CPUGPUMemory* DNNLayer::GetOutput()
{
	return output;
}

CPUGPUMemory* DNNLayer::GetDeltaInput()
{
	return deltaInput;
}

CPUGPUMemory* DNNLayer::GetParams()
{
	return params;
}

void DNNLayer::MakeStep()
{
	if (params != NULL && dparams != NULL)
	{
		float sugStepSize = -1;
		if (stepSize < 0)
		{
			params->CopyGPUtoCPU();
			dparams->CopyGPUtoCPU();

			double ps = 0;
			double dps = 0;
			float *p = (float*)params->GetCPUMemory();
			float *dp = (float*)dparams->GetCPUMemory();
			for (int i = 0; i < params->GetSize(); i++)
			{
				ps += abs(p[i]);
				dps += abs(dp[i]);
			}

			sugStepSize = (float)((-stepSize) * ((float)(ps + 0.01f) / (dps + 0.01f)));
// * Go always relative - do not limit it *
//			if (sugStepSize > (-stepSize))
//			{
//				sugStepSize = (-stepSize);
//			}
		}

		int threadsPerBlock = 256;
		int numBlocks = (params->GetSize() + threadsPerBlock - 1) / threadsPerBlock;
		make_step_kernel<<<numBlocks, threadsPerBlock>>>(
			(float*)params->GetGPUMemory(), (float*)dparams->GetGPUMemory(), stepSize < 0 ? sugStepSize : stepSize, params->GetSize());
		WaitForGPUToFinish();
	}
}

void DNNLayer::RemoveDeltaInput()
{
	if (deltaInput != NULL)
	{
		delete deltaInput;
		deltaInput = NULL;
	}
}

void DNNLayer::SetTrainRun()
{
	trainRun = true;
}

void DNNLayer::SetTestRun()
{
	trainRun = false;
}
