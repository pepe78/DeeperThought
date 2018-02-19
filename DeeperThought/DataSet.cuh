#ifndef DATASETCUH
#define DATASETCUH

#include "DataBatch.cuh"

#include <vector>
#include <string>

using namespace std;

class DataSet
{
private:
	vector<DataBatch*> batches;
public:
	DataSet(string &dataFile, int _input_width, bool _input_float, int _output_width, bool _output_float, int batchSize);
	~DataSet();

	int GetNumBatches();
	int GetNumSamples();
	DataBatch* GetBatchNumber(int i);
};

#endif