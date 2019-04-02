import csv
import numpy as np
import os
import pandas as pd
from os import listdir
from os.path import isfile, join


def getImages(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def generateTrainingData(dir, imageFiles, class_name, output):
    groundtruth = pd.read_csv(dir + '\groundtruth.txt', header = None)
    columns = ['filepath', 'x1', 'y1', 'x2', 'y2', 'class_name']
    trainingData = pd.DataFrame(columns=columns)
    trainingData['x1'] = groundtruth.iloc[:len(groundtruth), 0]
    trainingData['y1'] = groundtruth.iloc[:len(groundtruth), 1]
    trainingData['x2'] = groundtruth.iloc[:len(groundtruth), 0] + groundtruth.iloc[:len(groundtruth), 2]
    trainingData['y2'] = groundtruth.iloc[:len(groundtruth), 1] + groundtruth.iloc[:len(groundtruth), 3]
    trainingData['class_name'] = class_name
    trainingData['filepath'] = imageFiles
    trainingData['filepath'] = dir + '\img\\' + trainingData['filepath'].astype(str)
    trainingData.to_csv(dir + '\\' + output, index = False, header = None)

def main():
    data = pd.read_csv('data.csv', header=None)
    for index, row in data.iterrows():
        print(row[0], row[1], row[2])
        generateTrainingData(row[0], getImages(row[0]+'\img'), row[1], row[2])


main()