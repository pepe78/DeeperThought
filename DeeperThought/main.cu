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

	if (argc != 9)
	{
		printf("./dt configFile trainFile testFile batchSize(integer) paramFile/null saveEveryNEpochs(integer) square/log wheremax/netflix/none\n");
		exit(-1);
	}

	string configFile = (string)argv[1];
	string trainFile = (string)argv[2];
	string testFile = (string)argv[3];
	string batchSizeStr = (string)argv[4];
	string paramFile = (string)argv[5];
	int batchSize = convertToInt(batchSizeStr);
	string saveEveryStr = (string)argv[6];
	string errorType = (string)argv[7];
	string whereMax = (string)argv[8];
	int saveEvery = convertToInt(saveEveryStr);
	DNN *dnn = new DNN(configFile, trainFile, testFile, batchSize, paramFile, saveEvery, errorType, whereMax);

	dnn->Train();

	delete dnn;
	ReleaseGPU();

    return 0;
}
