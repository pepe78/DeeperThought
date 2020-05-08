#ifndef DNNLAYERPREPROCESSCUH
#define DNNLAYERPREPROCESSCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerPreprocess : public DNNLayer
{
private:
	int x1, x2;
	int x1SamplePoints, x2SamplePoints;
	float minAngle, maxAngle, minStretch, maxStretch, minNoise, maxNoise;
	CPUGPUMemory *dom;
	CPUGPUMemory *tmpMem;
public:
	DNNLayerPreprocess(int _x1, int _x2, int _batchSize, float _minAngle, float _maxAngle, float _minStretch, float _maxStretch, float _minNoise, float _maxNoise, int _x1SamplePoints, int _x2SamplePoints);
	~DNNLayerPreprocess();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
