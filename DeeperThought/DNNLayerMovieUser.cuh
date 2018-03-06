#ifndef DNNLAYERMOVIEUSERCUH
#define DNNLAYERMOVIEUSERCUH

#include "DNNLayer.cuh"
#include "GPU.cuh"

class DNNLayerMovieUser : public DNNLayer
{
private:
	int numMovies, numUsers;
	int vectorWidth;
public:
	DNNLayerMovieUser(int _numMovies, int _numUsers, int _vectorWidth, float _initValues, float _stepSize, int _batchSize);
	~DNNLayerMovieUser();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif