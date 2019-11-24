# -*- coding: utf-8 -*-
"""part2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wJY1Kcx25VhcBMX0TTXc4guCJgh56w1s
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
import pdb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler


MNIST_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])
MNIST_train = datasets.MNIST('.', download=True, train = True, transform=MNIST_transform)
MNIST_test = datasets.MNIST('.', download=True, train = False, transform=MNIST_transform)
FASHION_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2859], [0.3530])
])

FASHION_train = datasets.FashionMNIST('.', download=True, train=True, transform=MNIST_transform)
FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=FASHION_transform)

class GridDataset(Dataset):
    def __init__(self, MNIST_dataset, FASHION_dataset): # pass in dataset
        assert len(MNIST_dataset) == len(FASHION_dataset)
        self.MNIST_dataset, self.FASHION_dataset = MNIST_dataset, FASHION_dataset
        self.targets = FASHION_dataset.targets
        torch.manual_seed(442) # Fix random seed for reproducibility
        N = len(MNIST_dataset)
        self.randpos = torch.randint(low=0,high=4,size=(N,)) # position of the FASHION-MNIST image
        self.randidx = torch.randint(low=0,high=N,size=(N,3)) # indices of MNIST images
    
    def __len__(self):
        return len(self.MNIST_dataset)
    
    def __getitem__(self,idx): # Get one Fashion-MNIST image and three MNIST images to make a new image
        idx1, idx2, idx3 = self.randidx[idx]
        x = self.randpos[idx]%2
        y = self.randpos[idx]//2
        p1 = self.FASHION_dataset.__getitem__(idx)[0]
        p2 = self.MNIST_dataset.__getitem__(idx1)[0]
        p3 = self.MNIST_dataset.__getitem__(idx2)[0]
        p4 = self.MNIST_dataset.__getitem__(idx3)[0]
        combo = torch.cat((torch.cat((p1,p2),2),torch.cat((p3,p4),2)),1)
        combo = torch.roll(combo, (x*28,y*28), dims=(0,1))
        return (combo,self.targets[idx])
trainset = GridDataset(MNIST_train, FASHION_train)
# valset = GridDataset(MNIST_val, FASHION_val)
testset = GridDataset(MNIST_test, FASHION_test)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Design your own base module, define layers here
        out_channel = 24 # TODO: Put the output channel number of your base module here
        self.base = nn.Sequential(
            # nn.Conv2d(1,32,5,1,2),
            # nn.ReLU(),

            nn.Conv2d(1, 12, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(12, 48, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),

            nn.Conv2d(48, 24, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),

            nn.Conv2d(24, out_channel, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout()

        )
        # out_channel = 32 # TODO: Put the output channel number of your base module here
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channel,10)
        self.conv = nn.Conv2d(out_channel,10,1) # 1x1 conv layer (substitutes fc)

    def transfer(self): # Copy weights of fc layer into 1x1 conv layer
        self.conv.weight = nn.Parameter(self.fc.weight.unsqueeze(2).unsqueeze(3))
        self.conv.bias = nn.Parameter(self.fc.bias)

    def visualize(self,x):
        x = self.base(x)
        x = self.conv(x)
        return x
        
    def forward(self,x):
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
device = "cuda" if torch.cuda.is_available() else "cpu"
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
model = Network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # TODO: Experiment with different optimizer
num_epoch = 10 # TODO: Choose an appropriate number of epochs

loss_history = []


def train(model, loader, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
            loss_history.append(np.mean(running_loss))
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        # for batch, label in tqdm(loader):
        old_idx =10000
        for idx, data in enumerate(loader):
            batch, label = data
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            # if label == pred and idx < old_idx:
            #     vis_idx=idx 
            #     old_idx = idx
            if idx ==0:
                bool_list=torch.argmax(pred,dim=1)==label
                bool_idx_list=[i for i, x in enumerate(bool_list) if x]
                vis_idx = bool_idx_list[0]
                vis_label=label[vis_idx]
                print(torch.argmax(pred,dim=1))
                print(label)




            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc,vis_idx, vis_label

train(model, trainloader)

print("Evaluate on test set")
a,vis_idx,vis_label=evaluate(model, testloader)


model.transfer() # Copy the weights from fc layer to 1x1 conv layer

# TODO: Choose a correctly classified image and visualize it


vis_img=testset[vis_idx]
# vis_img=testset[4]
squeezed_vis=vis_img[0].unsqueeze(0)
squeezed_vis=squeezed_vis.to(device)
output=model.visualize(squeezed_vis)
output = output.squeeze(0)

print("Output vis shape {}".format(output.shape))
output_vis = output.cpu()
output_vis = output.softmax(dim=0).cpu()
output_vis= output_vis.detach().numpy()



print('corrent_label',vis_label)

original_img=squeezed_vis.cpu()
original_img= original_img.squeeze(0)
plt.imshow(original_img[0])
plt.colorbar()
plt.imsave('1.jpg',original_img[0])
plt.show()


fig, axs =plt.subplots(2,5)



for i in range(10):
    h = i//5
    w = i % 5
    # axs[h, w].imshow(output_vis[i], vmin=0, vmax=20, cmap='Greys_r')
    axs[h, w].imshow(output_vis[i], vmin=0, vmax=1)
    # axs[h, w].imshow(output_vis[i])
    axs[h, w].set_title('class {}'.format(i))


# plt.imshow(output_vis)

plt.show()
fig.savefig('classes.jpg')

# plt.imshow(output_vis[0], vmin=0, vmax=20, cmap='Greys_r')
plt.imshow(output_vis[vis_label], vmin=0, vmax=1)
plt.colorbar()
plt.show()



fig=plt.plot(loss_history,label='train loss')
    # plt.plot(val_acc_history,label='vaidation accuracy')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.show()






