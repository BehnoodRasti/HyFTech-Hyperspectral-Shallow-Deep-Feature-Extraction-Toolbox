import os
import numpy as np
import random
import torch.nn.functional as F
from torch.autograd import Variable
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

savepath = '/home/hrl/PycharmProjects/untitled/Hyperspectral/Data/FixedTrainSam/W3-DLSection/HU2013/CRNN-0.mat'

patchsize = 16  # input spatial size for 2D-CNN
batchsize = 64  # select from [16, 32, 64, 128], the best is 64
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
temp = x[:, :, 0]
pad_width = np.floor(patchsize/2)
pad_width = np.int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2,n2] = temp2.shape
x2 = np.empty((m2, n2, l), dtype='float32')

for i in range(l):
    temp = x[:, :, i]
    pad_width = np.floor(patchsize/2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x2[:, :, i] = temp2

# construct the training and testing set
[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch = np.empty((TrainNum, l, patchsize, patchsize), dtype='float32')
TrainLabel = np.empty(TrainNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
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

# ## data-augmentation
# TrainPatch1 = np.zeros_like(TrainPatch)
# TrainPatch2 = np.zeros_like(TrainPatch)
# TrainPatch3 = np.zeros_like(TrainPatch)
# TrainPatch4 = np.zeros_like(TrainPatch)
# TrainPatch5 = np.zeros_like(TrainPatch)
#
# for i in range(TrainPatch.shape[0]):
#     for j in range(TrainPatch.shape[1]):
#         TrainPatch1[i, j, ...] = np.rot90(TrainPatch[i, j, ...], 1)
#         TrainPatch2[i, j, ...] = np.rot90(TrainPatch[i, j, ...], 2)
#         TrainPatch3[i, j, ...] = np.rot90(TrainPatch[i, j, ...], 3)
#         TrainPatch4[i, j, ...] = np.flipud(TrainPatch[i, j, ...])
#         TrainPatch5[i, j, ...] = np.fliplr(TrainPatch[i, j, ...])
#
#
# TrainPatch = np.concatenate((TrainPatch, TrainPatch1, TrainPatch2, TrainPatch3, TrainPatch4, TrainPatch5), 0)
# TrainLabel = np.concatenate((TrainLabel, TrainLabel, TrainLabel, TrainLabel, TrainLabel, TrainLabel), 0)

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


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f + 1.)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.CLSTM1 = ConvLSTM(input_size=(patchsize, patchsize), input_dim=1, hidden_dim=[OutChannel],
                               kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.CLSTM2 = ConvLSTM(input_size=(patchsize//2, patchsize//2), input_dim=OutChannel, hidden_dim=[OutChannel*2],
                               kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.fc = nn.Linear(2*l*OutChannel, Classes)
        self.pool = nn.MaxPool2d(2)
        self.apool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        fx = torch.unsqueeze(x, 2)
        fo, fc = self.CLSTM1(fx)
        fo = fo[0].view(fo[0].size(0), l*OutChannel, patchsize, patchsize)
        fo = self.pool(fo)
        fo = fo.view(fo.size(0), l, OutChannel, patchsize//2, patchsize//2)
        fo, fc = self.CLSTM2(fo)
        fo = fo[0].view(fo[0].size(0), 2*l*OutChannel, patchsize//2, patchsize//2)
        fo = self.apool(fo)
        out = fo.view(fo.size(0), -1)
        out = self.fc(out)
        return out


cnn = Network()
print('The structure of the designed network', cnn)

# display variable name and shape
# for param_tensor in cnn.state_dict():
#     print(param_tensor, "\t", cnn.state_dict()[param_tensor].size())

cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_fun = nn.CrossEntropyLoss()  # the target label is not one-hotted

BestAcc = 0
# train and test the designed model
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

        # move train data to GPU
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output = cnn(b_x)
        cnn.zero_grad()
        loss = loss_fun(output, b_y)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            cnn.eval()

            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 50
            for i in range(number):
                temp = TestPatch[i * 50:(i + 1) * 50, :, :, :]
                temp = temp.cuda()
                temp2 = cnn(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 50:(i + 1) * 50] = temp3.cpu()
                del temp, temp2, temp3

            if (i + 1) * 50 < len(TestLabel):
                temp = TestPatch[(i + 1) * 50:len(TestLabel), :, :, :]
                temp = temp.cuda()
                temp2 = cnn(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 50:len(TestLabel)] = temp3.cpu()
                del temp, temp2, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
            # test_output = rnn(TestData)
            # pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            # accuracy = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)
            print('Epoch: ', epoch, '| loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            # save the parameters in network
            if accuracy > BestAcc:
                torch.save(cnn.state_dict(), 'net_params_AMTCNN_HS.pkl')
                BestAcc = accuracy

            cnn.train()


# # test each class accuracy
# # divide test set into many subsets

cnn.load_state_dict(torch.load('net_params_AMTCNN_HS.pkl'))
cnn.eval()

pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel)//50
for i in range(number):
    temp = TestPatch[i*50:(i+1)*50, :, :]
    temp = temp.cuda()
    temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i*50:(i+1)*50] = temp3.cpu()
    del temp, temp2, temp3

if (i+1)*50 < len(TestLabel):
    temp = TestPatch[(i+1)*50:len(TestLabel), :, :]
    temp = temp.cuda()
    temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i+1)*50:len(TestLabel)] = temp3.cpu()
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
part = 50
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

plt.figure()
plt.imshow(pred_all)
plt.show()




