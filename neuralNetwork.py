import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(flush_secs=1)


# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Network(nn.Module):

    def __init__(self, inSize, hidSize, outSize):
        super(Network, self).__init__()
        self.hidLyr = nn.Sequential(nn.Linear(inSize, hidSize), nn.LeakyReLU())
        self.outLyr = nn.Linear(hidSize, outSize)

    def forward(self, inputs):
        hid = self.hidLyr(inputs)
        out = self.outLyr(hid)
        return out


def runTraining(trainData):
    # Define hidden layer size, embedding size, and number of training epochs
    hidSize = 100
    inSize = 50
    outSize = 300
    epochs = 1

    # Initialize NNLM
    model = Network(inSize, hidSize, outSize)
    # model.to(dev)
    # Define the optimizer as Adam
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # Define the loss function as negative log likelihood loss
    criterion = nn.MSELoss()

    # Run training for specified number of epochs
    print('\nTraining...\n')
    for epoch in range(epochs):
        meanLoss = 0
        for i in range(len(trainData)):
            input = torch.Tensor(trainData[i][1])
            target = torch.Tensor(trainData[i][0])

            # Run model on input, get loss, update weights
            output = model(input)
            loss = criterion(output, target)
            #print(output, target, criterion(output, target))
            meanLoss += loss.item()
            writer.add_scalar("Loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch", epoch, "loss", meanLoss / len(trainData))
    # model.cpu()
    writer.close()
    return model
