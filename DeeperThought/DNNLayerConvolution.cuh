#ifndef DNNLAYERCONVOLUTIONCUH
#define DNNLAYERCONVOLUTIONCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerConvolution : public DNNLayer
{
private:
	int x1, x2, y1, y2;
	int numConvolutions;
	int numPics;
public:
	DNNLayerConvolution(int _numPics, int _x1, int _x2, int _numConvolutions, int _y1, int _y2, int _batchSize, float _initVal, float _stepSize);
	~DNNLayerConvolution();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif