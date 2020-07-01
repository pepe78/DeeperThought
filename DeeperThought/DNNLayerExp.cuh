#ifndef DNNLAYEREXPCUH
#define DNNLAYEREXPCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerExp : public DNNLayer
{
private:
public:
	DNNLayerExp(int _inputWidth, int _batchSize);
	~DNNLayerExp();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
