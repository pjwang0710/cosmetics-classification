import os

dataDir = "dataset"
logDir = "log"
modelDir = "model"

# for split_train_test
annotationFile = os.path.join(dataDir, "annotation.csv")
splitRate = 0.8
randomSeed = 100

# for train and evaluate
trainFilePath = os.path.join(dataDir, "train.csv")
testFilePath = os.path.join(dataDir, "test.csv")
imageFolder = os.path.join(dataDir, "cosmetics-all")
trainingPath = os.path.join(logDir, "train-output.txt")
testingPath = os.path.join(logDir, "test-output.txt")
saveModelPath = os.path.join(modelDir, "model.pth")
loadModelPath = ""

# parameters
inputSize = 224
batchSize = 32
numWorkers = 4
numClasses = 7
learningRate = 0.001
momentum = 0.9
usePretrained = True
epochs = 21
logInterval = 5
