#ifndef DNNLAYERRORCUH
#define DNNLAYERRORCUH

#include "CPUGPUMemory.cuh"

class DNNLayerError
{
private:
	CPUGPUMemory *error;
	CPUGPUMemory *deltaInput;

	int inputWidth, batchSize;
public:
	DNNLayerError(int _inputWidth, int _batchSize);
	~DNNLayerError();

	double ComputeError(CPUGPUMemory* output, CPUGPUMemory *expectedOutput);
	CPUGPUMemory* GetDeltaInput();
};

#endif