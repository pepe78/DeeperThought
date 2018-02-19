#ifndef DNNLAYERMAXCUH
#define DNNLAYERMAXCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerMax : public DNNLayer
{
private:
	int x1, x2, d1, d2;
	int numPics;
public:
	DNNLayerMax(int _numPics, int _x1, int _x2, int _d1, int _d2, int _batchSize);
	~DNNLayerMax();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif