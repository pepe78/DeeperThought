# DeeperThought

To run training:

DeeperThought.exe configFile trainFile testFile batchSize(integer) paramFile/null saveEveryNEpochs(integer)

## Input format:

expOut_1, ... , expOut_n, inp_1, ... , inp_m

For both trainFile and testFile (expOut - expected output, inp - input). One data point is one line.

## Results:

### configA.txt (logistic regression)

> matrix,784,10,0.5,0.0001

> sigmoid,10

![graphA](./results/graph_A.png)

Accuracy: 92.47 % (on test data)

### configB.txt (simple 2 layered network with dropout)

> matrix,784,100,0.5,0.001

> sigmoid,100

> dropout,100,0.25

> matrix,100,10,0.5,0.0001

> sigmoid,10

![graphB](./results/graph_B.png)

Accuracy: 96.24 % (on test data)
