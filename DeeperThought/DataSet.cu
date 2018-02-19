#include "DataSet.cuh"

DataSet::DataSet(string &dataFile, int _input_width, bool _input_float, int _output_width, bool _output_float, int batchSize)
{
	int numSamples = 0;
	ifstream is(dataFile);
	if (is.is_open())
	{
		while (true)
		{
			DataBatch *db = new DataBatch(is, _input_width, _input_float, _output_width, _output_float, batchSize);
			if (db->GetNumSamples() == 0)
			{
				break;
			}
			batches.push_back(db);
			numSamples += db->GetNumSamples();
			if (db->GetNumSamples() < batchSize)
			{
				break;
			}
		}
	}
	else
	{
		fprintf(stderr, "can't find file!\n");
		exit(-1);
	}

	is.close();

	printf("File: %s\n", dataFile.c_str());
	printf("Number of samples: %d\n", numSamples);
	printf("Number of batches: %d\n", (int)batches.size());
}

DataSet::~DataSet()
{
	for (size_t i = 0; i < batches.size(); i++)
	{
		delete batches[i];
	}
	batches.clear();
}

int DataSet::GetNumBatches()
{
	return (int)batches.size();
}

DataBatch* DataSet::GetBatchNumber(int i)
{
	return batches[i];
}

int DataSet::GetNumSamples()
{
	int ret = 0;
	for (size_t i = 0; i < batches.size(); i++)
	{
		ret += batches[i]->GetNumSamples();
	}

	return ret;
}
