
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from dataset import FacadeDataset

N_CLASS=5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        # self.layers = nn.Sequential(
        #     #########################################
        #     ###        TODO: Add more layers      ###
        #     #########################################
        #     # nn.Conv2d(3, self.n_class, 1, padding=0),
        #     # nn.ReLU(inplace=True)




        # )


        # self.down1=nn.Sequential(
        self.conv0= nn.Conv2d(3, 64, 3, padding=1)

        # down 1
        self.pool1=nn.MaxPool2d(2)
        self.conv1=nn.Conv2d(64, 128, 3, padding=1)
        self.batchnorm1 =  nn.BatchNorm2d(128)
        self.relu=nn.ReLU()

        # down 2
        self.pool2=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(128, 128, 3, padding=1)
        self.batchnorm2 =  nn.BatchNorm2d(128)
        self.relu=nn.ReLU()    

        # )

        # up 1
        self.up1=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv3 = nn.Conv2d(192, 128, 3, padding=1)
        self.conv_up1 = nn.Conv2d(256, 64, 3, padding=1)

        # up 2
        self.up2=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = nn.Conv2d(128, 64, 3, padding=1)
        self.outconv=nn.Conv2d(64, self.n_class, kernel_size=3, padding=1)



    def forward(self, x):
        # x = self.layers(x)
        x0 = self.conv0(x) # torch.Size([1, 64, 256, 256])

        # down1
        x1 = self.pool1(x0)
        x1 = self.conv1(x1)
        x1 = self.batchnorm1(x1)
        x1 = self.relu(x1) # torch.Size([1, 128, 128, 128])
  
        # down2
        x2 = self.pool2(x1) #torch.Size([1, 128, 64, 64])
        x2 = self.conv2(x2) #torch.Size([1, 128, 64, 64])
        x2 = self.batchnorm2(x2)
        x2 = self.relu(x2)

        # up1
        x = self.up1(x2) #torch.Size([1, 128, 128, 128])
        x = torch.cat([x,x1],dim=1) #torch.Size([1, 256, 128, 128])
        x = self.conv_up1 (x) #torch.Size([1, 64, 128, 128])

        # up2
        x = self.up2(x) #torch.Size([1, 64, 256, 256])
        x = torch.cat([x,x0],dim=1) #torch.Size([1, 128, 256, 256])
        x = self.conv_up2(x)

        logit = self.outconv(x)


        # x1cs= self.up1(x1c)
        # x2 = torch.cat([x0,x1cs],dim=1)
        # x3 = self.conv3(x2)
        # x4 = self.outconv(x3)


        return logit


def save_label(label, path):
    '''
    Function for ploting labels.
    '''
    colormap = [
        '#000000',
        '#0080FF',
        '#80FF80',
        '#FF8000',
        '#FF0000',
    ]
    assert(np.max(label)<len(colormap))
    colors = [hex2rgb(color, normalise=False) for color in colormap]
    w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
    with open(path, 'wb') as f:
        w.write(f, label)

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))
    return running_loss

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    return (losses/cnt)


def cal_AP(testloader, net, criterion, device):
    '''
    Calculate Average Precision
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        preds = [[] for _ in range(5)]
        heatmaps = [[] for _ in range(5)]
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images).cpu().numpy()
            for c in range(5):
                preds[c].append(output[:, c].reshape(-1))
                heatmaps[c].append(labels[:, c].cpu().numpy().reshape(-1))

        aps = []
        for c in range(5):
            preds[c] = np.concatenate(preds[c])
            heatmaps[c] = np.concatenate(heatmaps[c])
            if heatmaps[c].max() == 0:
                ap = float('nan')
            else:
                ap = ap_score(heatmaps[c], preds[c])
                aps.append(ap)
            print("AP = {}".format(ap))

    # print(losses / cnt)
    return None


def get_result(testloader, net, device, folder='output_train'):
    result = []
    cnt = 1
    with torch.no_grad():
        net = net.eval()
        cnt = 0
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)[0].cpu().numpy()
            c, h, w = output.shape
            assert(c == N_CLASS)
            y = np.zeros((h,w)).astype('uint8')
            for i in range(N_CLASS):
                mask = output[i]>0.5
                y[mask] = i
            gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
            save_label(y, './{}/y{}.png'.format(folder, cnt))
            save_label(gt, './{}/gt{}.png'.format(folder, cnt))
            plt.imsave(
                './{}/x{}.png'.format(folder, cnt),
                ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))

            cnt += 1

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('GPU USED?',torch.cuda.is_available())
    # TODO change data_range to include all train/evaluation/test data.
    # TODO adjust batch_size.
    train_data = FacadeDataset(flag='train', data_range=(0,885), onehot=False)
    train_loader = DataLoader(train_data, batch_size=1)

    val_data = FacadeDataset(flag='train', data_range=(885,905), onehot=False)
    val_loader = DataLoader(val_data, batch_size=1)


    test_data = FacadeDataset(flag='test_dev', data_range=(114,115), onehot=False)
    test_loader = DataLoader(test_data, batch_size=1)
    ap_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=True)
    ap_loader = DataLoader(ap_data, batch_size=1)

    name = 'starter_net'
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss() #TODO decide loss
    optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)

    train_loss_history=[]
    val_loss_history=[]

    test(train_loader, net, criterion, device)
    print('\nStart training')
    for epoch in range(5): #TODO decide epochs
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        train_loss=train(train_loader, net, criterion, optimizer, device, epoch+1)
        # TODO create your evaluation set, load the evaluation set and test on evaluation set
        evaluation_loader = val_loader
        val_loss=test(evaluation_loader, net, criterion, device)

        val_loss_history.append(val_loss)
        train_loss_history.append(train_loss)


    print('\nFinished Training, Testing on test set')
    test(test_loader, net, criterion, device)
    print('\nGenerating Unlabeled Result')
    result = get_result(test_loader, net, device, folder='output_test')

    torch.save(net.state_dict(), './models/model_{}.pth'.format(name))

    cal_AP(ap_loader, net, criterion, device)


    fig =plt.figure()
    plt.plot(train_loss_history,label='train loss')
    plt.plot(val_loss_history,label='vaidation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    fig.savefig('loss.jpg')


    # image_bbb
    # output = net(images)[0].cpu().numpy()

    # fig =plt.figure()
    # plt.plot(val_loss_history,label='vaidation loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # list_idx = [ i for i in range(10)]
    # plt.xticks(np.array(list_idx))
    # plt.legend()
    # plt.show()
    # fig.savefig('val_loss.jpg')

if __name__ == "__main__":
    main()
