# library
# standard library

import matplotlib.pyplot as plt
import numpy   as np
import pandas as pd
# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# load da
trainXPath = "../../kaggleDatasets/train_x.csv"
trainYPath = "../../kaggleDatasets/train_y.csv"

dtype = torch.cuda.FloatTensor
# dtype =  torch.FloatTensor
class kaggleDataset(Dataset):
    def __init__(self, csv_pathX, csv_pathY, transforms=None):
        self.x_data = pd.read_csv(csv_pathX)
        self.y_data = pd.read_csv(csv_pathY).as_matrix()
        self.transforms = transforms

    def __getitem__(self, index):
        # label = np.zeros((10))
        # label[self.y_data[index][0]] = 1
        # singleLable = torch.from_numpy(label).type(dtype)
        singleLable = torch.from_numpy(self.y_data[index]).type(torch.FloatTensor)
        singleX = np.asarray(self.x_data.iloc[index]).reshape(1, 64, 64)
        x_tensor = torch.from_numpy(singleX).type(dtype)
        return x_tensor, singleLable

    def __len__(self):
        return len(self.x_data.index)


# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 100
LR = 0.00001  # learning rate


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 64, 64)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 64, 64)
            nn.ReLU(),  # activation
            nn.MaxPool2d(  # reduce the size
                kernel_size=2,  # F
                stride=2  # W = (W-F)/S+1
            ),  # output shape (32, 16 , 16)
            #  choose max value in 2x2 area, output shape (16, 32, 32)
        )
        self.conv2 = nn.Sequential(  # input shape (1, 14, 14)
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.ReLU(),  # activation
            nn.MaxPool2d(  # reduce the size
                kernel_size=2,  # F
                stride=2  # W = (W-F)/S+1
            ),  # output shape (32, 16 , 16)
        )
        self.conv3 = nn.Sequential(  # input shape (1, 14, 14)
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.ReLU(),  # activation
        )
        self.out = nn.Linear(64 * 16 * 16, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = x.type(torch.cuda.DoubleTensor)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 16 * 16)
        output = self.out(x)
        return output


 # ,pin_memory=True)


def imgShower(data, target, numberOfExample):
    data = data.numpy()
    target = target.numpy()
    print(data.shape, target.shape)
    for i in range(numberOfExample):
        plt.title('Label is {label}'.format(label=target[i]))
        plt.imshow(data[i], cmap='gray')
        plt.show()


if __name__ == '__main__':
    trainData = kaggleDataset(trainXPath, trainYPath)
    train_loader = DataLoader(dataset=trainData, batch_size= BATCH_SIZE, shuffle=False)#, num_workers=1,pin_memory=True)
    # cnn = CNN().cuda()
    cnn = CNN().cuda()
    print(cnn)
    cnn.double()
    cnn.train()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    # loss_func = nn.MultiLabelSoftMarginLoss()
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    for epoch in range(EPOCH):
        for batch_idx, (data, target) in enumerate(train_loader):

            # imgShower(data,target)
            target = target.numpy()
            target = np.transpose(target)[0]
            data, target = Variable(data.type(dtype)), Variable(torch.from_numpy(target).type(dtype).long())
            output = cnn(data)  # cnn output
            # print(output.shape,target.shape)
            loss = loss_func(output, target)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))





# print 10 predictions from test data
# test_output, _ = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')
