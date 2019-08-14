#ifndef DNNLAYERAUGMENTMATRIXCUH
#define DNNLAYERAUGMENTMATRIXCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerAugmentMatrix : public DNNLayer
{
private:
	int x1, x2;
	int numConvolutions;
	int numPics;
public:
	DNNLayerAugmentMatrix(int _numPics, int _x1, int _x2, int _numConvolutions, int _batchSize, float _initVal, float _stepSize);
	~DNNLayerAugmentMatrix();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
