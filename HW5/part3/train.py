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

import torch.nn.functional as F
import unet

N_CLASS=5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        # self.layers = nn.Sequential(
            #########################################
            ###        TODO: Add more layers      ###
            #########################################
            # nn.Conv2d(3, self.n_class, 1, padding=0),
            # nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        
        self.unpool5 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool4 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool2 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool1 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)

        self.deconv5_1 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv5_2 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv5_3 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv4_1 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv4_2 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv4_3 = nn.ConvTranspose2d(512, 256, kernel_size = 3, stride = 1, padding = 1)
        self.deconv3_1 = nn.ConvTranspose2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.deconv3_2 = nn.ConvTranspose2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.deconv3_3 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 1, padding = 1)
        self.deconv2_1 = nn.ConvTranspose2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.deconv2_2 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1, padding = 1)
        self.deconv1_1 = nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.deconv1_2 = nn.ConvTranspose2d(64, self.n_class, kernel_size = 3, stride = 1, padding = 1)

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
        # )


        self.convlayers1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
            )

        self.convlayers2 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)

            )



        # self.convlayers3 = nn.Sequential(
        #     x = self.conv3_1(x)
        #     x = self.batch_norm3(x)
        #     x = F.relu(x)
        #     x = self.conv3_2(x)
        #     x = self.batch_norm3(x)
        #     x = F.relu(x)
        #     x = self.conv3_3(x)
        #     x = self.batch_norm3(x)
        #     x = F.relu(x)
        #     x, idxs3 = self.pool3(x)

        #     )


    def forward(self, x):
        # x = self.layers(x)

        size_1 = x.size()
        # x = self.conv1_1(x)
        # x = self.batch_norm1(x)
        # x = F.relu(x)
        # x = self.conv1_2(x)
        # x = self.batch_norm1(x)
        # x = F.relu(x)
        # x, idxs1 = self.pool1(x)

        x, idxs1  = self.convlayers1(x)
        
        
        size_2 = x.size()
        # x = self.conv2_1(x)
        # x = self.batch_norm2(x)
        # x = F.relu(x)
        # x = self.conv2_2(x)
        # x = self.batch_norm2(x)
        # x = F.relu(x)
        # x, idxs2 = self.pool2(x)

        x, idxs2  = self.convlayers2(x)
        
        size_3 = x.size()
        x = self.conv3_1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x, idxs3 = self.pool3(x)
        
        size_4 = x.size()
        x = self.conv4_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x, idxs4 = self.pool4(x)
        
        size_5 = x.size()
        x = self.conv5_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x, idxs5 = self.pool5(x)

        
        x = self.unpool5(x, idxs5, output_size = size_5)
        x = self.deconv5_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv5_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv5_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        
        x = self.unpool4(x, idxs4, output_size = size_4)
        x = self.deconv4_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4_3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        
        x = self.unpool3(x, idxs3, output_size = size_3)
        x = self.deconv3_1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3_2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3_3(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        
        x = self.unpool2(x, idxs2, output_size = size_2)
        x = self.deconv2_1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.deconv2_2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        x = self.unpool1(x, idxs1, output_size = size_1)
        x = self.deconv1_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.deconv1_2(x)
        return x


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
        # images = torch.unsqueeze(images, 1) # unsqueeze
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

# def main():
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# TODO change data_range to include all train/evaluation/test data.
# TODO adjust batch_size.
train_data = FacadeDataset(flag='train', data_range=(0,20), onehot=False)
train_loader = DataLoader(train_data, batch_size=1)

val_data = FacadeDataset(flag='train', data_range=(805,825), onehot=False)
val_loader = DataLoader(val_data, batch_size=1)


test_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=False)
test_loader = DataLoader(test_data, batch_size=1)
ap_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=True)
ap_loader = DataLoader(ap_data, batch_size=1)

name = 'starter_net'
# net = Net().to(device)
model = pytorch_unet.UNet(6)
net = model().to(device)

criterion = nn.CrossEntropyLoss() #TODO decide loss
optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)

train_loss_history=[]
val_loss_history=[]


print('\nStart training')
for epoch in range(1): #TODO decide epochs
    print('-----------------Epoch = %d-----------------' % (epoch+1))
    train_loss=train(train_loader, net, criterion, optimizer, device, epoch+1)
    # TODO create your evaluation set, load the evaluation set and test on evaluation set
    # evaluation_loader = train_loader
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
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
fig.savefig('train_loss.jpg')

fig =plt.figure()
plt.plot(val_loss_history,label='vaidation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
list_idx = [ i for i in range(20)]
plt.xticks(np.array(list_idx))
plt.legend()
plt.show()
fig.savefig('val_loss.jpg')




# if __name__ == "__main__":
#     main()
