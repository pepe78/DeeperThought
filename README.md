# Deeper Thought

Deeper Thought is general framework for training Deep Neural Networks (and convolutional as well).

To compile on linux:

> ./compile.sh

To compile on Windows with Visual Studio Community:

Open DeeperThought.sln file in VS and build.

To run training using your train and test data from command line execute:

> ./dt configFile trainFile testFile batchSize(integer) paramFile/null saveEveryNEpochs(integer) square/log wheremax/netflix/none

## Input format of data:

expOut_1, ... , expOut_n, inp_1, ... , inp_m

For both trainFile and testFile (expOut - expected output, inp - input). One data point is one line.

## Good research publications on the topic:

* N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer), JMLR, 2014

* J. Masci, U. Meier, D. Ciresan, and J. Schmidhuber: [Stacked Convolutional Auto-Encoders for
Hierarchical Feature Extraction](https://www.researchgate.net/profile/Jonathan_Masci/publication/221078713_Stacked_Convolutional_Auto-Encoders_for_Hierarchical_Feature_Extraction/links/0deec518b9c6ed4634000000/Stacked-Convolutional-Auto-Encoders-for-Hierarchical-Feature-Extraction.pdf), Artificial Neural Networks and Machine Learning – ICANN, 2011

* H. Xiao, K. Rasul, R. Vollgra: [Fashion-MNIST: a Novel Image Dataset for
Benchmarking Machine Learning Algorithms](https://arxiv.org/pdf/1708.07747.pdf), Arxiv, 2017

## Results for other datasets (combined results with WideOpenThoughts):

* [Fashion MNIST results](https://github.com/pepe78/WideOpenThoughts/blob/master/README_fashion.md)
* [Adult dataset results](https://github.com/pepe78/WideOpenThoughts/blob/master/README_adult.md)

## Results for MNIST dataset:

You can download [MNIST dataset here](https://www.dropbox.com/s/9eoiignb7tlrr2u/mnist.zip?dl=0). Mnist dataset can be found in other formats on [Yann Lecun's web page](http://yann.lecun.com/exdb/mnist/index.html) or [Kaggle's web page](https://www.kaggle.com/c/digit-recognizer).

### configA.txt (logistic regression)

#### batch size 4000, squared error

> matrix,784,10,0.5,0.0001

> sigmoid,10

Number of parameters in network: 7,850

![graphA](./results/graph_A.png)

Top accuracy on test data: 92.47 %

### configC.txt (logistic regression)

#### batch size 4000, squared error, auto step size

> matrix,784,10,0.5,-0.001

> sigmoid,10

Number of parameters in network: 7,850

![graphC](./results/graphC.png)

Top accuracy on test data: 92.53 %

### configB.txt (simple 2 layered network with dropout)

#### batch size 4000, squared error

> matrix,784,100,0.5,0.001

> sigmoid,100

> dropout,100,0.25

> matrix,100,10,0.5,0.0001

> sigmoid,10

Number of parameters in network: 79,510

![graphB](./results/graph_B.png)

Top accuracy on test data: 96.24 %

### configD.txt (simple 2 layered network with dropout)

#### batch size 4000, squared error, auto step size

> matrix,784,100,0.5,-0.01

> sigmoid,100

> dropout,100,0.25

> matrix,100,10,0.5,-0.01

> sigmoid,10

Number of parameters in network: 79,510

![graphD](./results/graphD.png)

Top accuracy on test data: 96.33 %

### configD.txt

#### batch size 4000, log loss error, auto step size

> matrix,784,100,0.5,-0.01

> sigmoid,100

> dropout,100,0.25

> matrix,100,10,0.5,-0.01

> sigmoid,10

Number of parameters in network: 79,510

![graphDlog](./results/graphDlog.png)

Top accuracy on test data: 96.73 %

### configE.txt (convolutional neural network CNN)

#### batch size 1000, log loss error, auto step size

> convolution,1,28,28,200,5,5,0.5,-0.001

> max,200,24,24,8,8

> matrix,1800,100,0.5,-0.001

> sigmoid,100

> dropout,100,0.25

> matrix,100,10,0.5,-0.001

> sigmoid,10

Number of parameters in network: 186,110

![graphE](./results/graphE.png)

Top accuracy on test data: 99.13 %

### configF.txt (convolutional neural network CNN)

#### batch size 4000, log loss error, auto step size

> convolution,1,28,28,400,5,5,0.5,-0.001

> max,400,24,24,6,6

> matrix,6400,150,0.5,-0.001

> sigmoid,150

> dropout,150,0.25

> matrix,150,10,0.5,-0.001

> softmax,10

Number of parameters in network: 971,660

![graphF](./results/graphF2.png)

Top accuracy on test data: 99.36 %

### configG.txt (convolutional neural network CNN)

#### batch size 4000, log loss error, auto step size

> convolution,1,28,28,200,8,8,0.5,-0.001

> max,200,21,21,7,7

> matrix,1800,130,0.5,-0.001

> sigmoid,130

> dropout,130,0.5

> matrix,130,10,0.5,-0.001

> softmax,10

Number of parameters in network: 248,240

![graphG](./results/graphG2.png)

Top accuracy on test data: 99.41 %

### configK.txt (low parameter convolutional neural network CNN)

#### batch size 3500, log loss error, auto step size

> convolution,1,28,28,20,8,8,0.5,-0.001

> max,20,21,21,7,7

> matrix,180,10,0.5,-0.001

> softmax,10

Number of parameters in network: 3,090

![graphK](./results/graphK.png)

Top accuracy on test data: 98.70 %

### configL.txt ([exact match convolutional](https://www.frisky.world/2020/06/exact-match-convolution.html) neural network EMC-NN)

#### batch size 3500, log loss error, auto step size

> [convolutionsq](https://www.frisky.world/2020/06/exact-match-convolution.html),1,28,28,500,8,8,0.0,1.0,-0.001

> max,500,21,21,21,21

> matrix,500,200,0.1,-0.001

> sigmoid,200

> dropout,200,0.5

> matrix,200,10,0.1,-0.001

> softmax,10

Number of parameters in network: 134,210

![graphL](./results/graphL.png)

Top accuracy on test data: 98.68 %

### configM.txt

#### batch size 3500, log loss error, auto step size

> ln,784

> matrix,784,500,0.1,-0.001

> sigmoid,500

> dropout,500,0.5

> ln,500

> matrix,500,10,0.1,-0.001

> softmax,10

Number of parameters in network: 397,510

![graphM](./results/graphM.png)

Top accuracy on test data: 98.40 %

## What you need to download and install beforehand:

* [Ubuntu for developers](https://www.ubuntu.com/desktop/developers) or Windows with [Visual Studio Community with C++](https://visualstudio.microsoft.com/vs/community/)

* CUDA for running on GPU: [CUDA](https://developer.nvidia.com/cuda-zone)

* CUDA GPU card. The one used for MNIST training above is [Nvidia GeForce GTX 1080 Ti](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080-ti/)

## Other alternatives for training DNNs:

* [CNTK](https://github.com/Microsoft/CNTK) - Microsoft

* [Pytorch](https://pytorch.org/) - Facebook

* [TensorFlow](https://www.tensorflow.org/) - Google

* [WideOpenThoughts](https://github.com/pepe78/WideOpenThoughts) - same as DeeperThought, but implemented in C++ OpenCL so that AMD GPU cards could be used
