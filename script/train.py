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

# path
TRAIN_FILE_PATH = conf.trainFilePath
TEST_FILE_PATH = conf.testFilePath
IMAGE_FOLDER = conf.imageFolder
TRAINING_LOG = conf.trainingPath
TESTING_LOG = conf.testingPath
SAVE_MODEL_PATH = conf.saveModelPath
LOAD_MODEL_PATH = conf.loadModelPath
LOG_INTERVAL = conf.logInterval

# parameters
INPUT_SIZE = conf.inputSize
BATCH_SIZE = conf.batchSize
NUM_WORKERS = conf.numWorkers
NUM_CLASSES = conf.numClasses
LEARNING_RATE = conf.learningRate
MOMENTUM = conf.momentum
USE_PRETRAINED = conf.usePretrained
EPOCHS = conf.epochs

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


def train(net, trainLoader, testLoader, optimizer, criterion, epochs):
    net.train()
    testAccuracy = 0
    bestModel = net
    for i in range(epochs):
        totalLoss = 0
        accuracy = 0
        count = 0
        for x, label in trainLoader:
            x = x.to(device)
            label = label.to(device, dtype=torch.long)
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, label)
            _, predicted = torch.max(output.data, 1)
            count += len(x)
            accuracy += (predicted == label).sum().item()      
            totalLoss += loss.item()*len(label)
            loss.backward()
            optimizer.step()
        with open(TRAINING_LOG, "a+") as f:
            f.write("Epoch: {} \n".format(i))
            f.write("Training Loss: {} \n".format(totalLoss / count))
            f.write("Training Accuracy: {} \n\n".format(accuracy / count))
        if (i % LOG_INTERVAL == 0):
            tmpAccuracy = test(net, testLoader, criterion, i)
            if (tmpAccuracy > testAccuracy):
                testAccuracy = tmpAccuracy
                bestModel = net
    torch.save(bestModel, SAVE_MODEL_PATH)
    return net


def test(net, testLoader, criterion, epoch):
    net.eval()
    totalLoss = 0
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
    with open(TESTING_LOG, "a+") as f:
        f.write("Epoch: {} \n".format(epoch))
        f.write("Testing Loss: {} \n".format(totalLoss / count))
        f.write("Testing Accuracy: {} \n\n".format(accuracy / count))
    return (accuracy / count)


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


def set_parameter_requires_grad(model, featureExtracting):
    if featureExtracting:
        for param in model.parameters():
            param.requires_grad = False


def main():
    dataTransformsTrain = transforms.Compose([
     transforms.RandomResizedCrop(INPUT_SIZE),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataTransformsTest = transforms.Compose([
     transforms.Resize(INPUT_SIZE),
     transforms.CenterCrop(INPUT_SIZE),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainDatasets = image_dataset(TRAIN_FILE_PATH, IMAGE_FOLDER, dataTransformsTrain)
    testDatasets = image_dataset(TEST_FILE_PATH, IMAGE_FOLDER, dataTransformsTest)
    dataloadersTrain = torch.utils.data.DataLoader(trainDatasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataloadersTest = torch.utils.data.DataLoader(testDatasets, batch_size=BATCH_SIZE, shuffle=False)
    
    if USE_PRETRAINED:
        model = VGG16_pretrained_model(numClasses=NUM_CLASSES, featureExtract=True, usePretrained=True).to(device)
    else:
        model = VGG16_model(numClasses=NUM_CLASSES).to(device)
    
    if LOAD_MODEL_PATH != "":
        model = torch.load(LOAD_MODEL_PATH).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss().to(device)
    model_ft = train(model, dataloadersTrain, dataloadersTest, optimizer, criterion, EPOCHS)
    
if __name__ == '__main__':
    main()