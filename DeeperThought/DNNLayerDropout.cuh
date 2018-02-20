#ifndef DNNLAYERDROPOUTCUH
#define DNNLAYERDROPOUTCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerDropout : public DNNLayer
{
private:
	CPUGPUMemory *dom;
	float perc;
public:
	DNNLayerDropout(int _inputWidth, int _batchSize, float _perc);
	~DNNLayerDropout();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif