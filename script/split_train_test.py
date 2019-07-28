import argparse
import os
import sys

import pandas as pd
import numpy as np

import config as conf

ANNOTATION_FILE = conf.annotationFile
TRAIN_FILE_PATH = conf.trainFilePath
TEST_FILE_PATH = conf.testFilePath
SPLIT_RATE = conf.splitRate
RANDOM_SEED = conf.randomSeed

def train_test_split_df(data, randomSeed, splitRate):
    np.random.seed(randomSeed)
    for i, label in enumerate(data.label.unique()):
        randomList = list(range(len(labels.get_group(label))))
        np.random.shuffle(randomList)
        trainPart = labels.get_group(label).iloc[randomList][:int(len(randomList)*splitRate)]
        testPart = labels.get_group(label).iloc[randomList][int(len(randomList)*splitRate):]
        if (i == 0):
            trainAll = trainPart
            testAll = testPart
        else:
            trainAll = pd.concat([trainAll, trainPart], axis=0)
            testAll = pd.concat([testAll, testPart], axis=0)
    return trainAll, testAll


if __name__ == '__main__':
    data = pd.read_csv(ANNOTATION_FILE)
    labels = data.groupby("label")
    
    trainAll, testAll = train_test_split_df(data, RANDOM_SEED, SPLIT_RATE)
    
    trainAll.to_csv(TRAIN_FILE_PATH, index=False)
    testAll.to_csv(TEST_FILE_PATH, index=False)
