#ifndef DATABATCHCUH
#define DATABATCHCUH

#include "CPUGPUMemory.cuh"

#include <fstream>
#include <vector>
#include <string>

using namespace std;

class DataBatch
{
private:
	int input_width;
	int output_width;
	int numSamples;

	CPUGPUMemory *inputs;
	CPUGPUMemory *outputs;

	void FillOut(bool _output_float, void * mem_outp, int width, vector<string> &parts, int offset);
public:
	DataBatch(ifstream &input, int _input_width, bool _input_float, int _output_width, bool _output_float, int maxSamples);
	~DataBatch();

	int GetNumSamples();
	CPUGPUMemory* GetInputs();
	CPUGPUMemory* GetOutputs();
};

#endif
