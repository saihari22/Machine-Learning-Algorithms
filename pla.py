import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from csv import reader


def readData(input):
    with open(input, 'r') as f:
        data = [datum for datum in reader(f, delimiter=",")]
    data_array = np.asarray(data)
    #headers = data_array[0]
    training_data = np.asarray(data_array[0:], dtype=float)
    train_dataX = training_data[:, 0:0 - 1]
    train_dataY = training_data[:, 0 - 1]
    size = len(train_dataX)
    affineCol = np.ones(size)
    train_dataX = np.insert(train_dataX, 2, affineCol, axis=1)

    return train_dataX,train_dataY

def predict(weight, x):
    score=0
    score = np.dot(weight, x)
    if score > 0:
        return 1
    else:
        return -1

def visualize_scatter(df, feat1=0, feat2=1, labels=2, weights=[-1, -1, 1],
                      title=''):
    """
        Scatter plot feat1 vs feat2.
        Assumes +/- binary labels.
        Plots first and second columns by default.
        Args:
          - df: dataframe with feat1, feat2, and labels
          - feat1: column name of first feature
          - feat2: column name of second feature
          - labels: column name of labels
          - weights: [w1, w2, b]
    """

    # Draw color-coded scatter plot
    colors = pd.Series(['r' if label > 0 else 'b' for label in df[labels]])
    ax = df.plot(x=feat1, y=feat2, kind='scatter', c=colors)

    # Get scatter plot boundaries to define line boundaries
    xmin, xmax = ax.get_xlim()

    # Compute and draw line. ax + by + c = 0  =>  y = -a/b*x - c/b
    a = weights[0]
    b = weights[1]
    c = weights[2]

    def y(x):
        return (-a/b)*x - c/b

    line_start = (xmin, xmax)
    line_end = (y(xmin), y(xmax))
    line = mlines.Line2D(line_start, line_end, color='red')
    ax.add_line(line)


    if title == '':
        title = 'Scatter of feature %s vs %s' %(str(feat1), str(feat2))
    ax.set_title(title)

    plt.show()

def perceptron(dataX,dataY):
    w = np.zeros(3)
    wSet = np.empty(shape=[0, 3])
    flag=True
    while True:
        flag = True
        for i in range(0,len(dataX)):
            score = predict(w, dataX[i])
            label = dataY[i]
            if label * score <= 0:
                flag = False
                w+=np.multiply(label,dataX[i])
        wSet = np.vstack((wSet, w))
        if flag:
            return  wSet

def write_csv(out, weights):
    dataframe = pd.DataFrame(data=weights)
    dataframe = dataframe[[dataframe.columns[i] for i in [0,1,2]]]
    dataframe.to_csv(out,index=False,header=False)

def main():
    '''YOUR CODE GOES HERE'''
    dataX,dataY = readData(sys.argv[1])
    wSet = perceptron(dataX,dataY)
    #print(wSet)
    write_csv(sys.argv[2], wSet)

    data = pd.read_csv(sys.argv[1], header=None)
    #visualize_scatter(data, weights=wSet[-1])

if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()