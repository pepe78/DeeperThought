#ifndef DNNLAYERMAXCUH
#define DNNLAYERMAXCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerMax : public DNNLayer
{
public:
	DNNLayerMax(int _inputWidth, int _outputWidth, int _batchSize);
	~DNNLayerMax();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif