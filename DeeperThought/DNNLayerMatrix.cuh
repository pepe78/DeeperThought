#ifndef DNNLAYERMATRIXCUH
#define DNNLAYERMATRIXCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerMatrix : public DNNLayer
{
public:
	DNNLayerMatrix(int _inputWidth, int _outputWidth, int _batchSize, float _initVal, float _stepSize);
	~DNNLayerMatrix();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif