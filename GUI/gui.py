###################MEMORY###################
import os
import gc
import json

###################DATASET###################
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable

###################IMAGE###################
import cv2


###################PLOTTING###################

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

###################PYTORCH MODEL###################
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

###################DATA STRUCTURES###################
import numpy as np
from collections import Counter

###################MEMORY SETTINGS###################

save_dir_splitted = "" #location where to save the preprocessed dataset
data_dir = "" #location where the preprocessed dataset is (in general = save_dir_splitted)
models_dir ="" #the resuls will be saved here


###################TRAINING SETTINGS###################

mse_weight = 50 #weight of the MSE loss during the training: i.e. LOSS = Cross Entropy + mse_weight x MSE


###################GENERAL SETTINGS###################
SEED = 42
np.random.seed(SEED) 

#used for the validation and the test to load all the patches of the same image together
class HDAdatasetPatches(Dataset):
    def __init__(self, root_dir,patch_dir, labels, train = False, grayscale= False, fft = False):
        self.root_dir = root_dir #directory of the full size images
        self.patch_dir = patch_dir #directory of the parches
        self.labels = labels #dictionary containing the labels of the full size images
        self.train = train #is train?
        self.grayscale=grayscale #return grayscale?
        self.fft = fft #return fft?

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.root_dir)[idx] #retrive image name
        img_id = int(img_name.split("_")[0]) #retrive image id (i.e. first part of the file name)
        
        patches_list = os.listdir(self.patch_dir) #retrive the list of all the patches
        
        #filter the patches that refer only to the specific image id
        patches_image_list = list(filter(lambda name: int(name.split("-")[0]) == img_id, patches_list))
        
        all_patches = []
        
        #read the full image and convert it to RGB (for visualizartion)
        full_image = cv2.imread(os.path.join(self.root_dir,img_name))
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        
        #iterate over all the patches of the current image
        for patch in patches_image_list:
            patch_path = os.path.join(self.patch_dir, patch)
            
        #the code is quite complex but the idea is to create a list of patches with shape compatible with pytorch:
        #i.e. (batch size, channel, dimension 1, dimension 2) since apply the to_tensor trainsform to the list of patches
        #create some propblems with the shape.
        #In this case the batch size is 1 since during the validation/test the patches are passed one at a time to the network
            if self.grayscale or self.fft:
                image = cv2.imread(patch_path,cv2.IMREAD_GRAYSCALE)
                
                channels = 1
            
            else:
                image = cv2.imread(patch_path)
                channels = 3

            if self.train:
                image = image.astype(np.uint8) + np.random.normal(0, noise_amount, image.shape).astype(np.uint8)

            if self.fft:
                image = np.fft.fft2(image)
                fshift = np.fft.fftshift(image)
                magnitude_spectrum = 20*np.log(np.abs(fshift))
                image = magnitude_spectrum
            
            image = (image/255.).astype(np.float32)
            image = image.reshape(image.shape[0],image.shape[1],channels)
            image = np.transpose(image, (2, 0, 1))
            all_patches.append(image.reshape(1,image.shape[0],image.shape[1],image.shape[2]))

        all_patches=np.stack(all_patches)
        
        #one hot encodig with labels smoothing
        label = 0.1*np.ones(3)
        label[self.labels[img_name]] = 0.8
        
        #notice that the labels is only one and it's the same for al patches in the list
        return all_patches, label.reshape(1,label.shape[0]), full_image

### TEST DATASET
test_dir = os.path.join(data_dir,"test","images")#images
test_dir_patches = os.path.join(data_dir,"test","images_patch")#patches
test_labels_dir = os.path.join(data_dir,"test","labels","test.json")
with open(test_labels_dir) as f:
    test_labels = json.load(f)#labels


test_dataset = HDAdatasetPatches(test_dir,test_dir_patches,test_labels,train = False,
                          grayscale=False,fft=False)

### VALIDATION DATASET
val_dir = os.path.join(data_dir,"val","images")#images
val_dir_patches = os.path.join(data_dir,"val","images_patch")#patches
val_labels_dir = os.path.join(data_dir,"val","labels","val.json")
with open(val_labels_dir) as f:
    val_labels = json.load(f)#labels

val_dataset = HDAdatasetPatches(val_dir,val_dir_patches,val_labels,train = False,
                          grayscale=False,fft=False)

### TRAIN DATASET
train_dir = os.path.join(data_dir,"train","images")#images
train_dir_patches = os.path.join(data_dir,"train","images_patch")#patches
train_labels_dir = os.path.join(data_dir,"train","labels","train.json")
with open(train_labels_dir) as f:
    train_labels = json.load(f)#labels

train_dataset = HDAdatasetPatches(train_dir,train_dir_patches,train_labels,train = False,
                          grayscale=False,fft=False)

classes = ["CLL", "FL", "MCL"]
id_to_name = {0:"CLL",1:"FL",2:"MCL"}
name_to_id = {"CLL":0,"FL":1,"MCL":2}

def plot_classification(dataset, idx):
    
    """
      Function used to plot the classification for a sample using only the count of the prediction of the patches for each image

      Parameters
      ----------
        dataset : HDAdatasetPatches object
            Object containing the dataset

        idx : int
            index of the dataset to compute e plot the prediction 
    """
    
    fig = plt.figure(constrained_layout=True,figsize=(20,20))
    gs = fig.add_gridspec(2,10)
    gs2 = gs[1,:].subgridspec(4,10)

    pred=[]
    labels = torch.tensor(dataset[idx][1]).to(device)
    true_image_label = labels.to("cpu").detach().numpy().argmax(axis=1)[0]

    count = {0:0,1:0,2:0}

    for i,patch in enumerate(dataset[idx][0]):

        ax = fig.add_subplot(gs2[i])


        img = torch.tensor(patch).to(device)

        rec_img,pred_labels= autoencoder(img)  


        curr = pred_labels.to("cpu").detach().numpy().argmax(axis=1)[0]

        count[curr] = count[curr]+1

        pred.append(curr)

        ax.set_title(" pred: "+r"$\bf{" + id_to_name[curr]  + "}$",fontsize=15)

        imagergb = cv2.cvtColor(patch.reshape(3,128,128).transpose(1,2,0), cv2.COLOR_BGR2RGB)
        ax.imshow(imagergb)
        ax.set_xticks([])
        ax.set_yticks([])

        if(true_image_label == curr):
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(4)  # change width
                ax.spines[axis].set_color('green')    # change color
        else:
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(4)  # change width
                ax.spines[axis].set_color('red')    # change color

    final_image_label = max(set(pred), key=pred.count) 


    ax = fig.add_subplot(gs[0,:])
    ax.set_title("true: "+r"$\bf{" + id_to_name[true_image_label]  + "}$"+ " pred: "+r"$\bf{" + id_to_name[final_image_label]  + "}$", fontsize=15)
    ax.imshow(dataset[idx][2])

    ax.text(dataset[ids][2].shape[1]+15,15,"Patches count:", fontsize=20,weight='bold')
    ax.text(dataset[ids][2].shape[1]+15,50,"  "+id_to_name[0]+": "+str(count[0]), fontsize=20)
    ax.text(dataset[ids][2].shape[1]+15,85,"  "+id_to_name[1]+": "+str(count[1]), fontsize=20)
    ax.text(dataset[ids][2].shape[1]+15,120,"  "+id_to_name[2]+": "+str(count[2]), fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

    if(true_image_label == final_image_label):
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(8)  # change width
            ax.spines[axis].set_color('green')    # change color
    else:
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(8)  # change width
            ax.spines[axis].set_color('red')    # change color
    plt.savefig("tmp.png")
    plt.close(fig)

def plot_classification_prob(dataset, idx,train_cm,th):
    
    """
      Function used to plot the classification for a sample using only the count of the prediction of the patches for each image 
      and the information provided by the train confusion matrix using the normal test

      Parameters
      ----------
        dataset : HDAdatasetPatches object
            Object containing the dataset

        idx : int
            index of the dataset to compute e plot the prediction

        train_cm: Numpy array
            Confusion matrix of the train dataset computed using the prediction provided by the test function

        th: int/float
            Paramenters that defined when two predictions are equiprobable
    """
    
    fig = plt.figure(constrained_layout=True,figsize=(20,20))
    gs = fig.add_gridspec(2,10)
    gs2 = gs[1,:].subgridspec(4,10)

    pred=[]
    labels = torch.tensor(dataset[idx][1]).to(device)
    true_image_label = labels.to("cpu").detach().numpy().argmax(axis=1)[0]

    count_weight = {0:0,1:0,2:0}
    count = {0:0,1:0,2:0}

    for i,patch in enumerate(dataset[idx][0]):

        ax = fig.add_subplot(gs2[i])


        img = torch.tensor(patch).to(device)

        rec_img,pred_labels= autoencoder(img)  
        
    

        weight = pred_labels.to("cpu").detach().numpy()
        for i in range(3):
            count_weight[i] = count_weight[i]+weight[0,i]

        curr = pred_labels.to("cpu").detach().numpy().argmax(axis=1)[0]
        count[curr] = count[curr] + 1
        
        pred.append(curr)


        string = ""
        for w in np.round(weight,2)[0]:
            string = string+str(w)+", "

        string = string[0:len(string)-2]
        ax.set_title(" pred: "+r"$\bf{"+id_to_name[curr]+"}$"+"\n weights: \n"+r"$\bf{" + string+"}$",fontsize=15)

        imagergb = cv2.cvtColor(patch.reshape(3,128,128).transpose(1,2,0), cv2.COLOR_BGR2RGB)
        ax.imshow(imagergb)
        ax.set_xticks([])
        ax.set_yticks([])

        if(true_image_label == curr):
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(4)  # change width
                ax.spines[axis].set_color('green')    # change color
        else:
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(4)  # change width
                ax.spines[axis].set_color('red')    # change color


    
    count_copy = count.copy()
    
    final_image_label = max(count, key=count.get) #extract the label with the max count
    max_value = count[final_image_label] #extract the count of that label

    del count[final_image_label] #remove the entry with the max count (now you can easily extract the second prediction
                                   # using the same procedure)*

    if len(count.keys())!=0: #*but only if there are other possible prediction (some samples are hard classified 
                            #,i.e. all 40 patches with the same predicted class)

        second_prediction = max(count, key=count.get)
        second_max_value = count[second_prediction]
        
        choice = [final_image_label,second_prediction]
        values = np.array([count_weight[final_image_label],count_weight[second_prediction]])

        #if second_max_value == max_value: #if the two counts are the same
        if(abs(count_weight[final_image_label]-count_weight[second_prediction])<=th):
            if(train_cm[second_prediction,final_image_label]>train_cm[final_image_label,second_prediction]):
                final_image_label = second_prediction
            else:
                final_image_label = final_image_label
        else:
            final_image_label = choice[values.argmax()]

       

    ax = fig.add_subplot(gs[0,:])
    ax.set_title("true: "+r"$\bf{" + id_to_name[true_image_label]  + "}$"+ "pred: "+r"$\bf{" + id_to_name[final_image_label]  + "}$", fontsize=15)
    ax.imshow(dataset[idx][2])

    ax.text(dataset[idx][2].shape[1]+15,15,"Patches count:", fontsize=20,weight='bold')
    ax.text(dataset[idx][2].shape[1]+15,50,"  "+id_to_name[0]+": "+str(count_copy[0]), fontsize=20)
    ax.text(dataset[idx][2].shape[1]+15,85,"  "+id_to_name[1]+": "+str(count_copy[1]), fontsize=20)
    ax.text(dataset[idx][2].shape[1]+15,120,"  "+id_to_name[2]+": "+str(count_copy[2]), fontsize=20)
    
    ax.text(dataset[idx][2].shape[1]+15,170,"Patches count weighted:", fontsize=20,weight='bold')
    ax.text(dataset[idx][2].shape[1]+15,205,"  "+id_to_name[0]+": "+str(count_weight[0]), fontsize=20)
    ax.text(dataset[idx][2].shape[1]+15,240,"  "+id_to_name[1]+": "+str(count_weight[1]), fontsize=20)
    ax.text(dataset[idx][2].shape[1]+15,275,"  "+id_to_name[2]+": "+str(count_weight[2]), fontsize=20)
    
    ax.text(dataset[idx][2].shape[1]+15,325,"Prediction parameters:", fontsize=20,weight='bold')
    ax.text(dataset[idx][2].shape[1]+15,360,"  Threshold: " +str(th), fontsize=20)
    ax.text(dataset[idx][2].shape[1]+15,395,
            "  Count from "+id_to_name[choice[0]]+" to "+id_to_name[choice[1]]+": "+str(train_cm[choice[0],choice[1]]),
            fontsize=20)
    ax.text(dataset[idx][2].shape[1]+15,430,
            "  Count from "+id_to_name[choice[1]]+" to "+id_to_name[choice[0]]+": "+str(train_cm[choice[1],choice[0]]),
            fontsize=20)
    
    ax.set_xticks([])
    ax.set_yticks([])

    if(true_image_label == final_image_label):
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(8)  # change width
            ax.spines[axis].set_color('green')    # change color
    else:
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(8)  # change width
            ax.spines[axis].set_color('red')    # change color
    
    plt.savefig("tmp.png")
    plt.close(fig)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        #encoder
        
        self.convE1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(0,0))
        self.convE2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(0,0))
        self.convE3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(0,0))
        self.flatten = nn.Flatten()
        self.BNE16 = nn.BatchNorm2d(16)
        self.BNE32 = nn.BatchNorm2d(32)
        self.BNE64 = nn.BatchNorm2d(64)
        self.linearE1 = nn.Linear(14400,50)
        self.linearE2 = nn.Linear(50,3)
        self.BNE150linear = nn.BatchNorm1d(50)
        
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.1)
        
        #decoder

        self.convD1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(0,0))
        self.convD2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(0,0))
        self.convD3 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(3,3), 
                                         stride=(2,2), padding=(0,0), dilation=1,output_padding=(1,1))
        self.reshape = nn.Unflatten(-1, (64,15,15))
        self.BND16 = nn.BatchNorm2d(16)
        self.BND32 = nn.BatchNorm2d(32)
        self.linearD1 = nn.Linear(3,50)
        self.linearD2 = nn.Linear(50,14400)
        self.BND150linear = nn.BatchNorm1d(50)
        self.BND16klinear = nn.BatchNorm1d(14400)
        
        self.out = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):

        x1 = self.act(self.BNE16(self.convE1(x)))
        x2 = self.act(self.BNE32(self.convE2(x1)))
        x3 = self.act(self.BNE64(self.convE3(x2)))
        x3linear = self.flatten(x3)
        x4 = self.drop(self.act(self.BNE150linear(self.linearE1(x3linear))))
        x5 = self.linearE2(x4)

        x = self.act(self.BND150linear(self.linearD1(x5)))
        x = self.act(self.BND16klinear(self.linearD2(x+x4)))
        x = self.reshape(x)
        x = self.act(self.BND32(self.convD1(x+x3)))
        x = self.act(self.BND16(self.convD2(x+x2)))
        x = self.out(self.convD3(x+x1))

        return x,self.softmax(x5)



cm = np.load("cm.npy",allow_pickle = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("You are working on: "+str(device))
autoencoder = Autoencoder().to(device)
autoencoder.load_state_dict(torch.load(os.path.join(models_dir,str(mse_weight)+"_best_ae_patches.pt"),map_location=device))
autoencoder.eval()

def on_trackbar1(val):
    global ids
    ids = val
    max_size = max(dataset[ids][2].shape[0],dataset[ids][2].shape[1])
    border = np.zeros((max_size,max_size))
    cv2.imshow("Classifier",cv2.cvtColor(dataset[ids][2], cv2.COLOR_RGB2BGR))

def on_trackbar2(val):
    global th
    th = val

def on_trackbar3(val):
    global type_test
    type_test = val

def on_trackbar4(val):
    global selector_dataset  
    selector_dataset = val
    menu_copy = np.copy(menu)
    text=""
    global dataset
    if(selector_dataset  == 0):
        text = "Train"
        
        dataset = train_dataset
        
    elif(selector_dataset  == 1):
        text="Validation"
        
        dataset = val_dataset
    else:
        text="Test"
        
        dataset=test_dataset
        
    menu_copy = cv2.putText(menu_copy, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("MENU",menu_copy)


font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
global selector_dataset
selector_dataset = 0
global ids
ids = 0
global th
th = 2
global type_test
type_test=0
global dataset
dataset = train_dataset


menu = np.zeros((1000,1000),np.float32)

while(True):
    
    cv2.namedWindow("MENU",cv2.WINDOW_FREERATIO)
    on_trackbar4(selector_dataset)
    cv2.createTrackbar("Dataset", "MENU" , selector_dataset, 2, on_trackbar4)
    key = cv2.waitKey(0)

    if(key==27):
        cv2.destroyAllWindows()
        break
        
    
    cv2.destroyWindow("MENU")
    cv2.namedWindow("Classifier",cv2.WINDOW_FREERATIO)
    cv2.createTrackbar("Sample ID", "Classifier" , ids, len(dataset)-1, on_trackbar1)
    cv2.createTrackbar("Threshold", "Classifier" , th, 40, on_trackbar2)
    cv2.createTrackbar("Weight?", "Classifier" , type_test, 1, on_trackbar3)
    on_trackbar1(ids)   
    key = 0



    while(True):
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            ids = 0
            break;
        else:
            if(type_test==0):
                plot_classification(dataset, ids)
            else:
                plot_classification_prob(dataset, ids,cm,th)

            result = cv2.imread("tmp.png")
            scale_percent = 50 # percent of original size
            width = int(result.shape[1] * scale_percent / 100)
            height = int(result.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            result = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
            cv2.namedWindow("RESULT",cv2.WINDOW_FREERATIO)
            cv2.imshow("RESULT",result)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                ids = 0
                break;
            else:
                cv2.destroyWindow("RESULT")
