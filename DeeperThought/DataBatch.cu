#include "DataBatch.cuh"
#include "StringUtils.cuh"

DataBatch::DataBatch(ifstream &input, int _input_width, bool _input_float, int _output_width, bool _output_float, int maxSamples)
{
	input_width = _input_width;
	output_width = _output_width;

	inputs = new CPUGPUMemory(_input_float, input_width * maxSamples, 0);
	outputs = new CPUGPUMemory(_output_float, output_width * maxSamples, 0);

	void* mem_inp = inputs->GetCPUMemory();
	void* mem_outp = outputs->GetCPUMemory();

	string line;
	numSamples = 0;
	while (getline(input, line))
	{
		vector<string> parts;
		split_without_space(line, parts, ',');
		if (parts.size() == 0)
		{
			break;
		}
		if (parts.size() != input_width + output_width)
		{
			fprintf(stderr, "wrong input size!\n");
			exit(-1);
		}

		FillOut(_output_float, mem_outp, output_width, parts, 0);
		FillOut(_input_float, mem_inp, input_width, parts, output_width);

		numSamples++;

		if (numSamples == maxSamples)
		{
			break;
		}
	}

	inputs->Resize(input_width * numSamples);
	outputs->Resize(output_width * numSamples);

	inputs->CopyCPUtoGPU();
	outputs->CopyCPUtoGPU();
}

DataBatch::~DataBatch()
{
	delete inputs;
	delete outputs;
}

void DataBatch::FillOut(bool _is_float, void *mem, int width, std::vector<string> &parts, int offset)
{
	if (_is_float)
	{
		float *p = (float*)mem;
		for (int i = 0; i < width; i++)
		{
			p[numSamples * width + i] = convertToFloat(parts[i + offset]);
		}
	}
	else
	{
		int *p = (int*)mem;
		for (int i = 0; i < width; i++)
		{
			p[numSamples * width + i] = convertToInt(parts[i + offset]);
		}
	}
}

int DataBatch::GetNumSamples()
{
	return numSamples;
}

CPUGPUMemory* DataBatch::GetInputs()
{
	return inputs;
}

CPUGPUMemory* DataBatch::GetOutputs()
{
	return outputs;
}
