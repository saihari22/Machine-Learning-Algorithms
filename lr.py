import numpy as np
import sys
from csv import reader
import pandas as pd

def readData(input):

    with open(input, 'r') as f:
        data = [datum for datum in reader(f, delimiter=",")]
    data_array = np.asarray(data)
    #headers = data_array[0]
    training_data = np.asarray(data_array[0:], dtype=float)
    train_dataX = training_data[:, :-1]
    train_dataY = training_data[:, [-1]]

    return train_dataX,train_dataY


def standardize(train_dataX):
    temp = np.copy(train_dataX)
    mu = np.mean(temp, axis=0)
    stdev = np.std(temp, axis=0)
    temp = (temp - mu) / stdev
    processed = np.ones((temp.shape[0], temp.shape[1] + 1))
    processed[:, 1:train_dataX.shape[1] + 1] = temp
    return processed

def GradientDescentAlgo(trainX, trainY, alpha, iterations=100):
    size = trainX.shape[0]
    betas = np.zeros((trainX.shape[1], 1))
    for i in range(0,iterations):
        temp=np.dot(trainX.transpose(), np.dot(trainX, betas) - trainY)
        betas = betas - (alpha/size)*temp
        sse = pow((np.linalg.norm(np.dot(trainX, betas) - trainY)),2)
    #print(alpha,sse)
    return (betas)


def writeCSV(wSet,iterations,alphas,outfile):
    iterationsDF = np.array([iterations] * len(alphas))
    alphas = np.array(alphas)
    dataframe = pd.DataFrame(data=wSet)
    dataframe[3] = alphas
    dataframe[4] = iterationsDF
    dataframe = dataframe[[dataframe.columns[i] for i in [3, 4, 0, 1, 2]]]
    dataframe.to_csv(outfile, index=False, header=False)

def main():
    """
    YOUR CODE GOES HERE
    Implement Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Write to an output csv file the outcome betas for each (alpha, iteration #) setting.
    Please run the file as follows: python3 lr.py data2.csv, results2.csv
    """
    trainX, trainY = readData(sys.argv[1])
    processed = standardize(trainX)
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.7] # 0.7 Chose as additional alpha value
    iterations = 100
    wSet = []
    for alpha in alphas:
        res = GradientDescentAlgo(processed, trainY, alpha, iterations)
        res = ['%.7f' % num for num in res]
        wSet.append((res))

    writeCSV(wSet,iterations,alphas,sys.argv[2])


if __name__ == "__main__":
    main()