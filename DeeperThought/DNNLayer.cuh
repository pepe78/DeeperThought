#ifndef DNNLAYERCUH
#define DNNLAYERCUH

#include "CPUGPUMemory.cuh"

class DNNLayer
{
protected:
	int inputWidth;
	int outputWidth;
	int numParams;

	float stepSize;

	CPUGPUMemory* deltaInput;
	CPUGPUMemory* output;
	CPUGPUMemory* params;
	CPUGPUMemory* dparams;

	bool trainRun;
public:
	DNNLayer(int _batchSize, int _inputWidth, int _outputWidth, int _numParams, float _initValMin, float _initValMax, float _stepSize);
	DNNLayer(int _batchSize, int _inputWidth, int _outputWidth, int _numParams, float _initVal, float _stepSize) : DNNLayer(_batchSize, _inputWidth, _outputWidth, _numParams, -_initVal, _initVal, _stepSize) {};
	~DNNLayer();

	virtual void Forward(CPUGPUMemory* input);
	virtual void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
	void ResetDeltaInput();

	int GetInputWidth();
	int GetOutputWidth();
	CPUGPUMemory* GetOutput();
	CPUGPUMemory* GetDeltaInput();
	CPUGPUMemory* GetParams();
	int GetNumParams();
	void MakeStep();

	void RemoveDeltaInput();

	void SetTrainRun();
	void SetTestRun();
};

#endif
