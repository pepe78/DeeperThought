#ifndef DNNLAYERMINCUH
#define DNNLAYERMINCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerMin : public DNNLayer
{
private:
	int x1, x2, d1, d2;
	int numPics;
public:
	DNNLayerMin(int _numPics, int _x1, int _x2, int _d1, int _d2, int _batchSize);
	~DNNLayerMin();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
