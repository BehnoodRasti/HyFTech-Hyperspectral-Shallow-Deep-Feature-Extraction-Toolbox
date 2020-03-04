import os
import numpy as np
import random

import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
from sklearn.decomposition import PCA

# setting parameters
DataPath = '/home/hrl/PycharmProjects/untitled/Hyperspectral/Data/FixedTrainSam/Houston/Houston.mat'
TRPath = '/home/hrl/PycharmProjects/untitled/Hyperspectral/Data/FixedTrainSam/Houston/TRLabel.mat'
TSPath = '/home/hrl/PycharmProjects/untitled/Hyperspectral/Data/FixedTrainSam/Houston/TSLabel.mat'

savepath = '/home/hrl/PycharmProjects/untitled/Hyperspectral/Data/FixedTrainSam/W3-DLSection/HU2013/2DCNN-14.mat'

patchsize = 16  # input spatial size for 2D-CNN
batchsize = 128  # select from [16, 32, 64, 128], the best is 64
EPOCH = 200
LR = 0.001


# load data
Data = io.loadmat(DataPath)
TrLabel = io.loadmat(TRPath)
TsLabel = io.loadmat(TSPath)

Data = Data['Houston']
Data = Data.astype(np.float32)
TrLabel = TrLabel['TRLabel']
TsLabel = TsLabel['TSLabel']


# without dimensionality reduction
pad_width = np.floor(patchsize/2)
pad_width = np.int(pad_width)

# normalization method 2: map to zero mean and one std
[m, n, l] = np.shape(Data)
# x2 = np.empty((m+pad_width*2, n+pad_width*2, l), dtype='float32')

for i in range(l):
    mean = np.mean(Data[:, :, i])
    std = np.std(Data[:, :, i])
    Data[:, :, i] = (Data[:, :, i] - mean)/std
    # x2[:, :, i] = np.pad(Data[:, :, i], pad_width, 'symmetric')

# # extract the first principal component
# x = np.reshape(Data, (m*n, l))
# pca = PCA(n_components=0.995, copy=True, whiten=False)
# x = pca.fit_transform(x)
# _, l = x.shape
# x = np.reshape(x, (m, n, l))
# # print x.shape
# # plt.figure()
# # plt.imshow(x)
# # plt.show()

x = Data

# boundary interpolation
temp = x[:,:,0]
pad_width = np.floor(patchsize/2)
pad_width = np.int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2,n2] = temp2.shape
x2 = np.empty((m2, n2, l), dtype='float32')

for i in range(l):
    temp = x[:,:,i]
    pad_width = np.floor(patchsize/2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x2[:,:,i] = temp2

# construct the training and testing set
[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch = np.empty((TrainNum, l, patchsize, patchsize), dtype='float32')
TrainLabel = np.empty(TrainNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    # patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
    patch = np.reshape(patch, (patchsize * patchsize, l))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (l, patchsize, patchsize))
    TrainPatch[i, :, :, :] = patch
    patchlabel = TrLabel[ind1[i], ind2[i]]
    TrainLabel[i] = patchlabel

[ind1, ind2] = np.where(TsLabel != 0)
TestNum = len(ind1)
TestPatch = np.empty((TestNum, l, patchsize, patchsize), dtype='float32')
TestLabel = np.empty(TestNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
    patch = np.reshape(patch, (patchsize * patchsize, l))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (l, patchsize, patchsize))
    TestPatch[i, :, :, :] = patch
    patchlabel = TsLabel[ind1[i], ind2[i]]
    TestLabel[i] = patchlabel

print('Training size and testing size are:', TrainPatch.shape, 'and', TestPatch.shape)

# step3: change data to the input type of PyTorch
TrainPatch = torch.from_numpy(TrainPatch)
TrainLabel = torch.from_numpy(TrainLabel)-1
TrainLabel = TrainLabel.long()
dataset = dataf.TensorDataset(TrainPatch, TrainLabel)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)

TestPatch = torch.from_numpy(TestPatch)
TestLabel = torch.from_numpy(TestLabel)-1
TestLabel = TestLabel.long()

Classes = len(np.unique(TrainLabel))

OutChannel = 32

# construct the network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = l,
                out_channels = OutChannel,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.BatchNorm2d(OutChannel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(OutChannel, OutChannel*2, 3, 1, 1),
            nn.BatchNorm2d(OutChannel*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(OutChannel*2, OutChannel*4, 3, 1, 1),
            nn.BatchNorm2d(OutChannel*4),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            # nn.Dropout(0.5),

        )

        self.out = nn.Linear(OutChannel*4, Classes)  # fully connected layer, output 16 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
print('The structure of the designed network', cnn)

# move model to GPU
cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

BestAcc = 0
# train and test the designed model
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

        # move train data to GPU
        b_x = b_x.cuda()
        b_y = b_y.cuda()

        output = cnn(b_x)  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            cnn.eval()

            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 5000
            for i in range(number):
                temp = TestPatch[i * 5000:(i + 1) * 5000, :, :, :]
                temp = temp.cuda()
                temp2 = cnn(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
                del temp, temp2, temp3

            if (i + 1) * 5000 < len(TestLabel):
                temp = TestPatch[(i + 1) * 5000:len(TestLabel), :, :, :]
                temp = temp.cuda()
                temp2 = cnn(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 5000:len(TestLabel)] = temp3.cpu()
                del temp, temp2, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
            # test_output = rnn(TestData)
            # pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            # accuracy = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            # save the parameters in network
            if accuracy > BestAcc:
                torch.save(cnn.state_dict(), 'W3-DLSection/HU2013/net_params_2DCNN.pkl')
                BestAcc = accuracy

            cnn.train()


# # test each class accuracy
# # divide test set into many subsets

cnn.load_state_dict(torch.load('W3-DLSection/HU2013/net_params_2DCNN.pkl'))
cnn.eval()

pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel)//5000
for i in range(number):
    temp = TestPatch[i*5000:(i+1)*5000, :, :]
    temp = temp.cuda()
    temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i*5000:(i+1)*5000] = temp3.cpu()
    del temp, temp2, temp3

if (i+1)*5000 < len(TestLabel):
    temp = TestPatch[(i+1)*5000:len(TestLabel), :, :]
    temp = temp.cuda()
    temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i+1)*5000:len(TestLabel)] = temp3.cpu()
    del temp, temp2, temp3

pred_y = torch.from_numpy(pred_y).long()
OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)

Classes = np.unique(TestLabel)
EachAcc = np.empty(len(Classes))

for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestLabel)):
        if TestLabel[j] == cla:
            sum += 1
        if TestLabel[j] == cla and pred_y[j] == cla:
            right += 1

    EachAcc[i] = right.__float__()/sum.__float__()


print(OA)
print(EachAcc)

del TestPatch, TrainPatch, TrainLabel, b_x, b_y, dataset, train_loader
# show the whole image
# The whole data is too big to test in one time; So dividing it into several parts
part = 5000
pred_all = np.empty((m*n, 1), dtype='float32')

number = m*n//part
for i in range(number):
    D = np.empty((part, l, patchsize, patchsize), dtype='float32')
    count = 0
    for j in range(i*part, (i+1)*part):
        row = j//n
        col = j - row*n
        row2 = row + pad_width
        col2 = col + pad_width
        patch = x2[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
        patch = np.reshape(patch, (patchsize * patchsize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, patchsize, patchsize))
        D[count, :, :, :] = patch
        count += 1

    temp = torch.from_numpy(D)
    temp = temp.cuda()
    temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_all[i*part:(i+1)*part, 0] = temp3.cpu()
    del temp, temp2, temp3, D

if (i+1)*part < m*n:
    D = np.empty((m*n-(i+1)*part, l, patchsize, patchsize), dtype='float32')
    count = 0
    for j in range((i+1)*part, m*n):
        row = j // n
        col = j - row * n
        row2 = row + pad_width
        col2 = col + pad_width
        patch = x2[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
        patch = np.reshape(patch, (patchsize * patchsize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, patchsize, patchsize))
        D[count, :, :, :] = patch
        count += 1

    temp = torch.from_numpy(D)
    temp = temp.cuda()
    temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_all[(i + 1) * part:m*n, 0] = temp3.cpu()
    del temp, temp2, temp3, D


pred_all = np.reshape(pred_all, (m, n)) + 1
OA = OA.numpy()
pred_y = pred_y.cpu()
pred_y = pred_y.numpy()
TestDataLabel = TestLabel.cpu()
TestDataLabel = TestDataLabel.numpy()

io.savemat(savepath, {'PredAll': pred_all, 'OA': OA, 'TestPre': pred_y, 'TestLabel': TestDataLabel})

# print io.loadmat(savepath)
#
plt.figure()
plt.imshow(pred_all)
plt.show()




