
import os
import numpy as np
import random

import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
from sklearn.decomposition import PCA

# using RNN with GRU for classification
# setting parameters
# setting parameters
DataPath = '/home/hrl/PycharmProjects/untitled/Hyperspectral/Data/FixedTrainSam/Houston/Houston.mat'
TRPath = '/home/hrl/PycharmProjects/untitled/Hyperspectral/Data/FixedTrainSam/Houston/TRLabel.mat'
TSPath = '/home/hrl/PycharmProjects/untitled/Hyperspectral/Data/FixedTrainSam/Houston/TSLabel.mat'

savepath = '/home/hrl/PycharmProjects/untitled/Hyperspectral/Data/FixedTrainSam/W3-DLSection/HU2013/AE-19.mat'

batchsize = 128
LR = 0.001
EPOCH = 200
EPOCH2 = 200

# load data
Data = io.loadmat(DataPath)
TrLabel = io.loadmat(TRPath)
TsLabel = io.loadmat(TSPath)

Data = Data['Houston']
Data = Data.astype(np.float32)
TrLabel = TrLabel['TRLabel']
TsLabel = TsLabel['TSLabel']

# normalization method 1: map to [0, 1]
[m, n, l] = Data.shape
for i in range(l):
    minimal = Data[:, :, i].min()
    maximal = Data[:, :, i].max()
    Data[:, :, i] = (Data[:, :, i] - minimal)/(maximal - minimal)

# transform data to matrix
TotalData = np.reshape(Data, [m*n, l])
TrainDataLabel = np.reshape(TrLabel, [m*n, 1])
Tr_index, _ = np.where(TrainDataLabel != 0)
TrainData1 = TotalData[Tr_index, :]
TrainDataLabel = TrainDataLabel[Tr_index, 0]
TestDataLabel = np.reshape(TsLabel, [m*n, 1])
Ts_index, _ = np.where(TestDataLabel != 0)
TestData1 = TotalData[Ts_index, :]
TestDataLabel = TestDataLabel[Ts_index, 0]

# construct data for network

TrainData = np.empty((len(TrainDataLabel), l, 1), dtype='float32')
TestData = np.empty((len(TestDataLabel), l, 1), dtype='float32')

for i in range(len(TrainDataLabel)):
    temp = TrainData1[i, :]
    temp = np.transpose(temp)
    TrainData[i, :, 0] = temp

for i in range(len(TestDataLabel)):
    temp = TestData1[i, :]
    temp = np.transpose(temp)
    TestData[i, :, 0] = temp

print('Training size and testing size are:', TrainData.shape, 'and', TestData.shape)

TrainData = torch.from_numpy(TrainData)
TrainDataLabel = torch.from_numpy(TrainDataLabel)-1
TrainDataLabel = TrainDataLabel.long()
dataset = dataf.TensorDataset(TrainData, TrainDataLabel)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)

TestData = torch.from_numpy(TestData)
TestDataLabel = torch.from_numpy(TestDataLabel)-1
TestDataLabel = TestDataLabel.long()

# train_loader for AE using training and test data together
TrainData2 = torch.cat([TrainData, TestData], 0)
TrainDataLabel2 = torch.cat([TrainDataLabel, TestDataLabel], 0)
dataset2 = dataf.TensorDataset(TrainData2, TrainDataLabel2)
train_loader2 = dataf.DataLoader(dataset2, batch_size=batchsize, shuffle=True)
print('Training size for reconstruction', TrainData2.shape)


Classes = len(np.unique(TrainDataLabel))

# construct the reconstruction network
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(TrainData.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, TrainData.shape[1]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


ae = AE()
ae.cuda()
print(ae)

optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
loss_func = nn.MSELoss()

BestAcc = 100

# train the AE and save the best model
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader2):  # gives batch data
        b_x = b_x.cuda()

        b_x = b_x.view(-1, TrainData.shape[1])
        output = ae(b_x)  # rnn output

        loss = loss_func(output, b_x)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())
            # save the parameters in network
            if loss.data.cpu().numpy() < BestAcc:
                torch.save(ae.state_dict(), 'W3-DLSection/HU2013/net_params_AERecons.pkl')
                BestAcc = loss.data.cpu().numpy()

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(TrainData.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),

            # nn.Linear(TrainData.shape[1], 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
        )

        self.output = nn.Linear(128, Classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.output(x)
        return x

# train classifier
classifier = Classifier()
classifier.cuda()
print(classifier)

optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# load pre-trained ae parameters
model_dict = classifier.state_dict()
pretrained_dict = torch.load('W3-DLSection/HU2013/net_params_AERecons.pkl')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
classifier.load_state_dict(model_dict)


BestAcc = 0

# training and testing
for epoch in range(EPOCH2):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = b_x.cuda()
        b_x = b_x.view(-1, TrainData.shape[1])
        b_y = b_y.cuda()
        output = classifier(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:

            classifier.eval()  # in the testing phase, we don't need to use dropout

            # divide test set into many subsets
            pred_y = np.empty((len(TestDataLabel)), dtype='float32')
            number = len(TestDataLabel) // 5000
            for i in range(number):
                temp = TestData[i * 5000:(i + 1) * 5000, :, :]
                temp = temp.cuda()
                temp = temp.view(-1, TrainData.shape[1])
                temp2 = classifier(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
                del temp, temp2, temp3

            if (i + 1) * 5000 < len(TestDataLabel):
                temp = TestData[(i + 1) * 5000:len(TestDataLabel), :, :]
                temp = temp.cuda()
                temp = temp.view(-1, TrainData.shape[1])
                temp2 = classifier(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 5000:len(TestDataLabel)] = temp3.cpu()
                del temp, temp2, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)
            # test_output = rnn(TestData)
            # pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            # accuracy = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

            if accuracy > BestAcc:
                torch.save(classifier.state_dict(), 'W3-DLSection/HU2013/net_params_AEClass.pkl')
                BestAcc = accuracy

            classifier.train()  # in the training phase, we need to use dropout again

# # test each class accuracy
# # divide test set into many subsets
# rnn.eval()
classifier.load_state_dict(torch.load('W3-DLSection/HU2013/net_params_AEClass.pkl'))
classifier.eval()
pred_y = np.empty((len(TestDataLabel)), dtype='float32')
number = len(TestDataLabel)//5000
for i in range(number):
    temp = TestData[i*5000:(i+1)*5000, :, :]
    temp = temp.cuda()
    temp = temp.view(-1, TrainData.shape[1])
    temp2 = classifier(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i*5000:(i+1)*5000] = temp3.cpu()
    del temp, temp2, temp3

if (i+1)*5000 < len(TestDataLabel):
    temp = TestData[(i+1)*5000:len(TestDataLabel), :, :]
    temp = temp.cuda()
    temp = temp.view(-1, TrainData.shape[1])
    temp2 = classifier(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i+1)*5000:len(TestDataLabel)] = temp3.cpu()
    del temp, temp2, temp3

pred_y = torch.from_numpy(pred_y).long()
OA = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)

Classes = np.unique(TestDataLabel)
EachAcc = np.empty(len(Classes))

for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestDataLabel)):
        if TestDataLabel[j] == cla:
            sum += 1
        if TestDataLabel[j] == cla and pred_y[j] == cla:
            right += 1

    EachAcc[i] = right.__float__()/sum.__float__()


print(OA)
print(EachAcc)

del TestData, TrainDataLabel, b_x, b_y, dataset, train_loader
# show the whole image
# The whole data is too big to test in one time; So dividing it into several parts
D = np.empty((m*n, l, 1), dtype='float32')
pred_all = np.empty((m*n, 1), dtype='float32')
count = 0
for i in range(m*n):
    temp = TotalData[i, :]
    temp = np.transpose(temp)
    D[count, :, 0] = temp
    count += 1

del temp
# D = torch.from_numpy(D)
number = m*n//5000
for i in range(number):
    temp = D[i*5000:(i+1)*5000, :, :]
    temp = torch.from_numpy(temp)
    temp = temp.cuda()
    temp = temp.view(-1, TrainData.shape[1])
    temp2 = classifier(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_all[i*5000:(i+1)*5000, 0] = temp3.cpu()
    del temp, temp2, temp3


if (i+1)*5000 < m*n:
    temp = D[(i+1)*5000:m*n, :, :]
    temp = torch.from_numpy(temp)
    temp = temp.cuda()
    temp = temp.view(-1, TrainData.shape[1])
    temp2 = classifier(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_all[(i+1)*5000:m*n, 0] = temp3.cpu()
    del temp, temp2, temp3


pred_all = np.reshape(pred_all, (m, n)) + 1
OA = OA.numpy()
pred_y = pred_y.cpu()
pred_y = pred_y.numpy()
TestDataLabel = TestDataLabel.cpu()
TestDataLabel = TestDataLabel.numpy()

io.savemat(savepath, {'PredAll': pred_all, 'OA': OA, 'TestPre': pred_y, 'TestLabel': TestDataLabel})

# print io.loadmat(savepath)
#
plt.figure()
plt.imshow(pred_all)
plt.show()