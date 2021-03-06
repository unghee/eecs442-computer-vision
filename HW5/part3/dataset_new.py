import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from scipy import signal, stats
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np



class EnableDataset(Dataset):
    def __init__(self, dataDir='./Data/' ,data_range=(1, 10), window_size=150, stride=500, delay=500, processed=False):

        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.dataset = []
        # self.img_data_stack=np.array([],shape=(51, 3, 4, 51), dtype=np.int64)
        self.img_data_stack=np.empty((51, 3, 4, 51), dtype=np.int64)
        subject_list= ['156','185']
        self.dataset = []
        for i in range(data_range[0], data_range[1]):   
        	raw_data = pd.read_csv(dataDir +'AB' + subject_list[0]+'/Processed/'+'AB' + subject_list[0]+ '_Circuit_%03d_post.csv'% i)


	        segmented_data = np.array([], dtype=np.int64).reshape(0,window_size,48)
	        labels = np.array([], dtype=np.int64)
	        timesteps = []
	        triggers = []
	        index = 0
	        while not pd.isnull(raw_data.loc[index, 'Right_Heel_Contact']):
	            timesteps.append(raw_data.loc[index, 'Right_Heel_Contact'])
	            trigger = raw_data.loc[index, 'Right_Heel_Contact_Trigger']
	            trigger=str(int(trigger))
	            triggers.append(trigger) # triggers can be used to compare translational and steady-state error
	            labels = np.append(labels,[float(trigger[2])], axis =0)
	            index += 1
	        index = 0
	        while not pd.isnull(raw_data.loc[index, 'Right_Toe_Off']):
	            timesteps.append(raw_data.loc[index, 'Right_Toe_Off'])
	            trigger = raw_data.loc[index, 'Right_Toe_Off_Trigger']
	            trigger=str(int(trigger))
	            triggers.append(trigger) # triggers can be used to compare translational and steady-state error
	            labels = np.append(labels,[float(trigger[2])], axis =0)
	            index += 1
	        index = 0
	        while not pd.isnull(raw_data.loc[index, 'Left_Heel_Contact']):
	            timesteps.append(raw_data.loc[index, 'Left_Heel_Contact'])
	            trigger = raw_data.loc[index, 'Left_Heel_Contact_Trigger']
	            trigger=str(int(trigger))
	            triggers.append(trigger) # triggers can be used to compare translational and steady-state error
	            labels = np.append(labels,[float(trigger[2])], axis =0)
	            index += 1
	        index = 0 
	        while not pd.isnull(raw_data.loc[index, 'Left_Toe_Off']):
	            timesteps.append(raw_data.loc[index, 'Left_Toe_Off'])
	            trigger = raw_data.loc[index, 'Left_Toe_Off_Trigger']
	            trigger=str(int(trigger))
	            triggers.append(trigger) # triggers can be used to compare translational and steady-state error
	            labels = np.append(labels,[float(trigger[2])], axis =0)
	            index += 1
	        index = 0 

	        for idx,timestep in enumerate(timesteps):
	            if timestep-window_size-1 >= 0:
	                # labels = np.append(labels, [raw_data.loc[timestep, 'Mode']], axis=0)
	                data = raw_data.loc[timestep-window_size-1:timestep-2, 'Right_Shank_Ax':'Left_Knee']
	                img= spectrogram2(data)
	                self.dataset.append((img,labels[idx]))
	    print("load dataset done")

    def spectrogram2(self, segmented_data, fs=500):
        vals1 = []
        for x in range(3):
            row = segmented_data[:,x]
            f, t, Sxx = signal.spectrogram(row, fs, window=signal.windows.hamming(100, True), noverlap=50)
            tmp, _ = stats.boxcox(Sxx.reshape(-1,1))
            Sxx = tmp.reshape(Sxx.shape)-np.min(tmp)
            Sxx = Sxx/np.max(Sxx)*255
            vals1.append(Sxx)
        vals2 = []
        for x in range(6,9):
            row = segmented_data[:,x]
            f, t, Sxx = signal.spectrogram(row, fs, window=signal.windows.hamming(100, True), noverlap=50)
            tmp, _ = stats.boxcox(Sxx.reshape(-1,1))
            Sxx = tmp.reshape(Sxx.shape)-np.min(tmp)
            Sxx = Sxx/np.max(Sxx)*255
            vals2.append(Sxx)

        out1 = np.stack(vals1, axis=2)
        out2 = np.stack(vals2, axis=2)
        out = np.hstack((out1, out2))

        out = np.flipud(out)

        out=out.astype(np.uint8)

    	return out