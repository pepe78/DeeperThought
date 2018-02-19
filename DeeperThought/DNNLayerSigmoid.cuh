#ifndef DNNLAYERSIGMOIDCUH
#define DNNLAYERSIGMOIDCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerSigmoid : public DNNLayer
{
public:
	DNNLayerSigmoid(int _inputWidth, int _batchSize);
	~DNNLayerSigmoid();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif