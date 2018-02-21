#ifndef DNNLAYERRELUCUH
#define DNNLAYERRELUCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerRelu : public DNNLayer
{
public:
	DNNLayerRelu(int _inputWidth, int _batchSize);
	~DNNLayerRelu();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif