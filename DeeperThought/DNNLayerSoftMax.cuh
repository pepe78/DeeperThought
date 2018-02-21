#ifndef DNNLAYERSOFTMAX
#define DNNLAYERSOFTMAX

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerSoftMax : public DNNLayer
{
public:
	DNNLayerSoftMax(int _inputWidth, int _batchSize);
	~DNNLayerSoftMax();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif