#ifndef DNNLAYERLNCUH
#define DNNLAYERLNCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerLn : public DNNLayer
{
private:
public:
	DNNLayerLn(int _inputWidth, int _batchSize);
	~DNNLayerLn();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
