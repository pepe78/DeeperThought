#include "DNN.cuh"

#include "StringUtils.cuh"
#include "DNNLayerConvolution.cuh"
#include "DNNLayerMatrix.cuh"
#include "DNNLayerMax.cuh"
#include "DNNLayerSigmoid.cuh"
#include "DNNLayerSoftMax.cuh"
#include "DNNLayerRelu.cuh"
#include "DNNLayerDropout.cuh"

#include <fstream>
#include <string>

using namespace std;

DNN::DNN(string &configFile, string &trainSetFile, string &testSetFile, int _batchSize, string &paramFile, int _saveEvery, string &errorType)
{
	saveEvery = _saveEvery;
	ifstream is(configFile);
	if (is.is_open())
	{
		string line;
		while (getline(is, line))
		{
			if (line.length() > 0)
			{
				printf("Layer %d: %s\n", (int)layers.size(), line.c_str());
				vector<string> parts;
				split_without_space(line, parts, ',');

				if (parts[0].compare("convolution") == 0)
				{
					if (parts.size() != 9)
					{
						fprintf(stderr, "wrong setup of convolution layer!\n");
						exit(-1);
					}
					int numPics = convertToInt(parts[1]);
					int x1 = convertToInt(parts[2]);
					int x2 = convertToInt(parts[3]);
					int numConvo = convertToInt(parts[4]);
					int y1 = convertToInt(parts[5]);
					int y2 = convertToInt(parts[6]);
					float initVal = convertToFloat(parts[7]);
					float stepSize = convertToFloat(parts[8]);

					DNNLayer *curLayer = new DNNLayerConvolution(numPics, x1, x2, numConvo, y1, y2, _batchSize, initVal, stepSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("matrix") == 0)
				{
					if (parts.size() != 5)
					{
						fprintf(stderr, "wrong setup of matrix layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);
					int outpWidth = convertToInt(parts[2]);
					float initVal = convertToFloat(parts[3]);
					float stepSize = convertToFloat(parts[4]);

					DNNLayer *curLayer = new DNNLayerMatrix(inpWidth, outpWidth, _batchSize, initVal, stepSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("sigmoid") == 0)
				{
					if (parts.size() != 2)
					{
						fprintf(stderr, "wrong setup of sigmoid layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);

					DNNLayer *curLayer = new DNNLayerSigmoid(inpWidth, _batchSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("relu") == 0)
				{
					if (parts.size() != 2)
					{
						fprintf(stderr, "wrong setup of relu layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);

					DNNLayer *curLayer = new DNNLayerRelu(inpWidth, _batchSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("softmax") == 0)
				{
					if (parts.size() != 2)
					{
						fprintf(stderr, "wrong setup of softmax layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);

					DNNLayer *curLayer = new DNNLayerSoftMax(inpWidth, _batchSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("dropout") == 0)
				{
					if (parts.size() != 3)
					{
						fprintf(stderr, "wrong setup of dropout layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);
					float perc = convertToFloat(parts[2]);

					DNNLayer *curLayer = new DNNLayerDropout(inpWidth, _batchSize, perc);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("max") == 0)
				{
					if (parts.size() != 6)
					{
						fprintf(stderr, "wrong setup of max layer!\n");
						exit(-1);
					}
					int numPics = convertToInt(parts[1]);
					int x1 = convertToInt(parts[2]);
					int x2 = convertToInt(parts[3]);
					int d1 = convertToInt(parts[4]);
					int d2 = convertToInt(parts[5]);

					DNNLayer *curLayer = new DNNLayerMax(numPics, x1, x2, d1, d2, _batchSize);
					layers.push_back(curLayer);
				}
				else
				{
					fprintf(stderr, "type of layer %s not implemented yet!\n", parts[0].c_str());
					exit(-1);
				}

				if (layers.size() > 1 && layers[layers.size() - 2]->GetOutputWidth() != layers[layers.size() - 1]->GetInputWidth())
				{
					fprintf(stderr, "outputs of layer does not match input of layer!\n");
					exit(-1);
				}
			}
		}
	}
	trainSet = new DataSet(trainSetFile, layers[0]->GetInputWidth(), true, layers[layers.size() - 1]->GetOutputWidth(), true, _batchSize);
	testSet = new DataSet(testSetFile, layers[0]->GetInputWidth(), true, layers[layers.size() - 1]->GetOutputWidth(), true, _batchSize);

	errorLayer = new DNNLayerError(layers[layers.size() - 1]->GetOutputWidth(), _batchSize, errorType.compare("square") == 0);

	epoch = 0;

	layers[0]->RemoveDeltaInput();
	LoadFromFile(paramFile);
}

DNN::~DNN()
{
	for (size_t i = 0; i < layers.size(); i++)
	{
		delete layers[i];
	}
	layers.clear();

	if (trainSet != NULL)
	{
		delete trainSet;
	}
	if (testSet != NULL)
	{
		delete testSet;
	}

	delete errorLayer;
}

void DNN::Train()
{
	while (true)
	{
		printf("Epoch %d \n", epoch);
		TrainEpoch();
		Test();
		SaveToFile();
		epoch++;
	}
}

void DNN::Forward(CPUGPUMemory *firstInput, bool isTrain)
{
	for (size_t i = 0; i < layers.size(); i++)
	{
		if (isTrain)
		{
			layers[i]->SetTrainRun();
		}
		else
		{
			layers[i]->SetTestRun();
		}
		layers[i]->Forward(i == 0 ? firstInput : layers[i - 1]->GetOutput());
		WaitForGPUToFinish();
	}
}

void DNN::BackWard(CPUGPUMemory *firstInput)
{
	for (int i = (int)layers.size() - 1; i >= 0; i--)
	{
		layers[i]->SetTrainRun();
		layers[i]->ResetDeltaInput();
		layers[i]->Backward(i == 0 ? firstInput : layers[i - 1]->GetOutput(), i == layers.size() - 1 ? errorLayer->GetDeltaInput() : layers[i + 1]->GetDeltaInput());
		WaitForGPUToFinish();

		layers[i]->MakeStep();
	}
}

int DNN::ComputeCorrect(CPUGPUMemory *expected_output, CPUGPUMemory *output)
{
	int ret = 0;
	int outputWidth = layers[layers.size() - 1]->GetOutputWidth();
	int numSamples = expected_output->GetSize() / outputWidth;

	expected_output->CopyGPUtoCPU();
	output->CopyGPUtoCPU();

	float* eo = (float*)expected_output->GetCPUMemory();
	float* o = (float*)output->GetCPUMemory();
	for (int i = 0; i < numSamples; i++)
	{
		int p1 = 0;
		for (int j = 1; j < outputWidth; j++)
		{
			if (eo[i * outputWidth + p1] < eo[i * outputWidth + j])
			{
				p1 = j;
			}
		}

		int p2 = 0;
		for (int j = 1; j < outputWidth; j++)
		{
			if (o[i * outputWidth + p2] < o[i * outputWidth + j])
			{
				p2 = j;
			}
		}
		if (p1 == p2)
		{
			ret++;
		}
	}

	return ret;
}

double DNN::TrainBatch(int batchNum)
{
	Forward(trainSet->GetBatchNumber(batchNum)->GetInputs(), true);
	double ret = errorLayer->ComputeError(layers[layers.size() - 1]->GetOutput(), trainSet->GetBatchNumber(batchNum)->GetOutputs());
	BackWard(trainSet->GetBatchNumber(batchNum)->GetInputs());

	return ret;
}

double DNN::TestBatch(int batchNum)
{
	Forward(testSet->GetBatchNumber(batchNum)->GetInputs(), false);
	return errorLayer->ComputeError(layers[layers.size() - 1]->GetOutput(), testSet->GetBatchNumber(batchNum)->GetOutputs());
}

void DNN::TrainEpoch()
{
	vector<int> batchesToDo;
	for (int i = 0; i < trainSet->GetNumBatches(); i++)
	{
		batchesToDo.push_back(i);
	}

	double ret = 0;
	int correct = 0;
	while(batchesToDo.size() !=0)
	{
		int which = rand() % batchesToDo.size();
		int bn = batchesToDo[which];
		batchesToDo.erase(batchesToDo.begin() + which);
		double curErr = TrainBatch(bn);
		ret += curErr;
		int curCorrect = ComputeCorrect(trainSet->GetBatchNumber(bn)->GetOutputs(), layers[layers.size() - 1]->GetOutput());
		correct += curCorrect;
		printf("Train Batch %d CurError %lf (%d) Error %lf (%d)\n", bn, curErr, curCorrect, ret, correct);
	}
	printf("TrainError %lf (%d)\n", ret, correct);
	string txt = convertToString(epoch) + ((string)",") + convertToString((float)ret / (trainSet->GetNumSamples() + 0.0f)) + ((string)",") + convertToString(correct / (trainSet->GetNumSamples() + 0.0f)) + ((string)",");
	AppendToFile("debug.csv", txt);
}

void DNN::Test()
{
	double ret = 0;
	int correct = 0;
	for (int i = 0; i < testSet->GetNumBatches(); i++)
	{
		double curErr = TestBatch(i);
		ret += curErr;
		int curCorrect = ComputeCorrect(testSet->GetBatchNumber(i)->GetOutputs(), layers[layers.size() - 1]->GetOutput());
		correct += curCorrect;
		printf("Test Batch %d CurError %lf (%d) Error %lf (%d)\n", i, curErr, curCorrect, ret, correct);
	}
	printf("TestError %lf (%d)\n", ret, correct);
	string txt = convertToString((float)ret / (testSet->GetNumSamples() + 0.0f)) + ((string)",") + convertToString(correct / (testSet->GetNumSamples()+ 0.0f)) + ((string)"\n");
	AppendToFile("debug.csv", txt);
}

void DNN::SaveToFile()
{
	if (epoch % saveEvery == 0)
	{
		string filename = "params_";
		filename += convertToString(epoch);
		filename += ".bin";
		ofstream os(filename, ios::out | ios::binary);
		for (size_t i = 0; i < layers.size(); i++)
		{
			CPUGPUMemory *m = layers[i]->GetParams();
			if (m != NULL)
			{
				m->SaveToFile(os);
			}
		}
		os.close();
	}
}

void DNN::LoadFromFile(string &paramFile)
{
	ifstream is(paramFile, ios::in | ios::binary);

	if (is.is_open())
	{
		printf("loading file %s\n", paramFile.c_str());
		for (size_t i = 0; i < layers.size(); i++)
		{
			CPUGPUMemory *m = layers[i]->GetParams();
			if (m != NULL)
			{
				m->LoadFromFile(is);
			}
		}
		is.close();

		string numEp = getNumbersOnly(paramFile);
		epoch = convertToInt(numEp) + 1;
	}
}