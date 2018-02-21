#ifndef DNNLAYERRORCUH
#define DNNLAYERRORCUH

#include "CPUGPUMemory.cuh"

class DNNLayerError
{
private:
	CPUGPUMemory *error;
	CPUGPUMemory *deltaInput;

	int inputWidth, batchSize;
	bool square;
public:
	DNNLayerError(int _inputWidth, int _batchSize, bool _square);
	~DNNLayerError();

	double ComputeError(CPUGPUMemory* output, CPUGPUMemory *expectedOutput);
	CPUGPUMemory* GetDeltaInput();
};

#endif