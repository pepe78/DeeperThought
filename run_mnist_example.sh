if [[ ! -f mnist.zip ]]
then
	wget https://www.dropbox.com/s/9eoiignb7tlrr2u/mnist.zip
	unzip mnist.zip
fi

./compile.sh

./dt ./results/configA.txt mnist_train.txt mnist_test.txt 4000 null 10 log wheremax
