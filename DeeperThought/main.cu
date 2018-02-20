#include "DNN.cuh"
#include "GPU.cuh"
#include "CPUGPUMemory.cuh"
#include "RandUtils.cuh"
#include "StringUtils.cuh"


#include <stdio.h>


int main(int argc, char *argv[])
{
	InitRand();
	InitGPU();

	if (argc != 7)
	{
		printf("DeeperThought.exe configFile trainFile testFile batchSize(integer) paramFile/null saveEveryNEpochs(integer)\n");
		exit(-1);
	}

	string configFile = (string)argv[1];
	string trainFile = (string)argv[2];
	string testFile = (string)argv[3];
	string batchSizeStr = (string)argv[4];
	string paramFile = (string)argv[5];
	int batchSize = convertToInt(batchSizeStr);
	string saveEveryStr = (string)argv[6];
	int saveEvery = convertToInt(saveEveryStr);
	DNN *dnn = new DNN(configFile, trainFile, testFile, batchSize, paramFile, saveEvery);

	for (int r = 0; r < 1000000; r++)
	{
		dnn->Train();
		dnn->Test();
	}

	delete dnn;
	ReleaseGPU();

    return 0;
}
