import argparse
import os

import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.models as models
import torchvision.transforms as transforms

import config as conf

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EVALUATE_FILE = conf.evaluateFile
TEST_FILE_PATH = conf.testFilePath
IMAGE_FOLDER = conf.imageFolder
LOAD_MODEL_PATH = conf.loadModelPath

INPUT_SIZE = conf.inputSize
BATCH_SIZE = conf.batchSize
NUM_WORKERS = conf.numWorkers
NUM_CLASSES = conf.numClasses
LEARNING_RATE = conf.learningRate
MOMENTUM = conf.momentum
USE_PRETRAINED = conf.usePretrained
EPOCHS = conf.epochs

def test(net, testLoader):
    net.eval()
    accuracy = 0
    count = 0
    for x, label in testLoader:
        x = x.to(device)
        label = label.to(device, dtype=torch.long)
        output = net(x)
        loss = criterion(output, label)
        _, predicted = torch.max(output.data, 1)
        count += len(x)
        accuracy += (predicted == label).sum().item()
        totalLoss += loss.item()*len(label)
    return (accuracy / count)


class VGG16_model(nn.Module):
    def __init__(self, numClasses=7):
        super(VGG16_model, self).__init__()
        self.vgg16 = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64,128,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numClasses)
        )

    def forward(self, x):
        x = self.vgg16(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def VGG16_pretrained_model(numClasses, featureExtract=True, usePretrained=True):
    model = models.vgg16(pretrained=True)
    set_parameter_requires_grad(model, featureExtract)
    numFtrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(numFtrs,numClasses)
    return model


class image_dataset(Dataset):
    def __init__(self, csvFile, rootPath, transform):
        df = pd.read_csv(csvFile)
        self.rootPath = rootPath
        self.xTrain = df['path']
        self.yTrain = pd.factorize(df['label'], sort=True)[0]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.rootPath, self.xTrain[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.yTrain[index]

    def __len__(self):
        return len(self.xTrain.index)


def main():
    dataTransformsTest = transforms.Compose([
     transforms.Resize(INPUT_SIZE),
     transforms.CenterCrop(INPUT_SIZE),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    testDatasets = image_dataset(TEST_FILE_PATH, IMAGE_FOLDER, dataTransformsTest)
    dataloadersTest = torch.utils.data.DataLoader(testDatasets, batch_size=BATCH_SIZE, shuffle=False)
    
    if USE_PRETRAINED:
        model = VGG16_pretrained_model(numClasses=NUM_CLASSES, featureExtract=True, usePretrained=True).to(device)
        if LOAD_MODEL_PATH != "":
            model = torch.load(LOAD_MODEL_PATH).to(device)
    else:
        model = VGG16_model(numClasses=NUM_CLASSES).to(device)
    
    accuracy = test(model, dataloadersTrain, dataloadersTest, optimizer, criterion, EPOCHS)
    with open(EVALUATE_FILE, "a+") as f:
        f.write("input size: {} \n".format(INPUT_SIZE))
        f.write("classes number: {} \n".format(NUM_CLASSES))
        f.write("use pretrained: {} \n".format(USE_PRETRAINED))
        f.write("epochs: {} \n".format(EPOCHS))
        f.write("batch size: {} \n".format(BATCH_SIZE))
        f.write("learning rate: {} \n".format(LEARNING_RATE))
        f.write("momentum: {} \n".format(MOMENTUM))
        f.write("accuracy: {} \n\n".format(accuracy))

    
if __name__ == '__main__':
    main()