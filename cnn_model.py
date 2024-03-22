# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:34:15 2023

@author: asus
"""



import sys
import time
import numpy as np
import pandas as pd 
import warnings
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from torch import optim 
import random
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import metrics as mc

import Model_Metrics as mmc

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 指定GPU or CPU 进行训练


bufs=np.load("Lym_dataset.npy") 

bufs1=np.load("testLym_dataset.npy") 

Xd=[]
Yd=[]
MASK=[]
POS=[]


org_imgs=[]
org_labels=[]

for em in bufs: 
    Xd.append(em[1])
    Yd.append(em[2])
    MASK.append(np.array([em[3],em[4]]))
    POS.append(np.array(em[5]))
    org_imgs.append(em[0])
    org_labels.append(em[2][1])

Xd1=[]
Yd1=[]
MASK1=[]
POS1=[]

for em in bufs1: 
    Xd1.append(em[1])
    Yd1.append(em[2])
    MASK1.append(np.array([em[3],em[4]]))
    POS1.append(np.array(em[5]))


fg=int(len(bufs)/2)

# for p in range(len(org_imgs[fg:])):  
#     plt.imsave('saved_figs/Tested_orgimgs/'+str(p)+'_ _.jpg',org_imgs[fg+p],cmap='gray')    
# np.savetxt('saved_figs/truths.txt',np.round(org_labels[fg:]))

# for p in range(len(org_imgs[fg:])):  
#     plt.imsave('saved_figs/International_Tested_orgimgs/'+str(p)+'_ _.jpg',org_imgs[fg+p],cmap='gray')    
# np.savetxt('saved_figs/truths.txt',np.round(org_labels[fg:]))


# seqs=random.sample(range(0,len(bufs)),len(bufs)) 
# np.save("seqs.npy",seqs) 
# seqs=np.load("seqs.npy")

seqs=np.arange(0,len(bufs),1) 

sel1=seqs[0:fg] 
sel2=seqs[fg:]   
X_train=torch.from_numpy(np.array(Xd)[sel1,:,:].astype(np.float32)).to(device)
X_vadit=torch.from_numpy(np.array(Xd)[sel2,:,:].astype(np.float32)).to(device)
X_test=torch.from_numpy(np.array(Xd1)[:,:,:].astype(np.float32)).to(device)

Y_train=torch.from_numpy(np.array(Yd)[sel1,:].astype(np.float32)).to(device)
Y_vadit=torch.from_numpy(np.array(Yd)[sel2,:].astype(np.float32)).to(device)
Y_test=torch.from_numpy(np.array(Yd1)[:,:].astype(np.float32)).to(device)


MASKS_train=torch.from_numpy(np.array(MASK)[sel1,:,:,:].astype(np.float32)).to(device) 
MASKS_vadit=torch.from_numpy(np.array(MASK)[sel2,:,:,:].astype(np.float32)).to(device)
MASKS_test=torch.from_numpy(np.array(MASK1)[:,:,:,:].astype(np.float32)).to(device)

pos_train=torch.from_numpy(np.array(POS)[sel1,:].astype(np.float32)).to(device) 
pos_vadit=torch.from_numpy(np.array(POS)[sel2,:].astype(np.float32)).to(device)
pos_test=torch.from_numpy(np.array(POS1)[:,:].astype(np.float32)).to(device)

train_bufs=bufs[:fg]
vadit_bufs=bufs[fg:]  
test_bufs=bufs1[:]              
                
Lx,Ly=np.meshgrid(np.linspace(0, 1, 80),np.linspace(0, 1, 80))
Lx,Ly=torch.from_numpy(Lx.astype(np.float32)).to(device),\
      torch.from_numpy(Ly.astype(np.float32)).to(device)
      
     

class MLCNN(nn.Module): # 集成nn.Module父类
    def __init__(self):
        super(MLCNN, self).__init__()# 看一下具体的参数
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1,bias=True)
        self.pool = nn.MaxPool2d(2, 2)#池化为原来大小的1/2
        self.conv2 = nn.Conv2d(16,32, 3,stride=1,padding=1,bias=True)#6个特征数 每个大小16*5
       
        #self.filter=torch.where()
    
        
        
        self.conv3 = nn.Conv2d(32,64, 3,stride=1,padding=1,bias=False)#6个特征数 每个大小16*5
        self.conv4 = nn.Conv2d(64,64, 3,stride=1,padding=1,bias=False)#6个特征数 每个大小16*5
        self.conv5 = nn.Conv2d(64,16, 3,stride=1,padding=1,bias=False)#6个特征数 每个大小16*5
        
        
        self.upsample1 = nn.ConvTranspose2d(16,8,3,stride=2, padding=1, output_padding=1)

        self.upsample2 = nn.ConvTranspose2d(8,1, 3,stride=2, padding=1, output_padding=1)
        
        #self.amplify=F.interpolate(scale_factor=4,mode='bicubic')#最邻近插值法

        self.Tfc1 = nn.Linear(12800 , 256)
        self.Tfc2 = nn.Linear(256, 64)
        self.Tfc3 = nn.Linear(64, 2)# self.relu = nn.ReLU(inplace=True)# 正向传播 10分类标签
        
        self.conv6 = nn.Conv2d(2,16, 3,stride=1,padding=1,bias=False)#6个特征数 每个大小16*5
        self.conv7 = nn.Conv2d(16,32, 3,stride=1,padding=1,bias=False)#6个特征数 每个大小16*5
        self.conv8 = nn.Conv2d(32,64,3,stride=1,padding=1,bias=False)#6个特征数 每个大小16*5
        self.conv9 = nn.Conv2d(64,256,3,stride=1,padding=1,bias=False)#6个特征数 每个大小16*5
        self.conv10 = nn.Conv2d(256,512,3,stride=1,padding=1,bias=False)#6个特征数 每个大小16*5
        

        
        
        self.Bfc1 = nn.Linear(2048 , pos_train.shape[1])
        
        self.Bfc2 = nn.Linear(pos_train.shape[1], pos_train.shape[1])
        # self.Bfc3 = nn.Linear(256, 14)# self.relu = nn.ReLU(inplace=True)# 正向传播 10分类标签
        
        
        
        # torch.nn.init.constant_(self.fc1.weight,val=0.001)
        # torch.nn.init.constant_(self.fc1.bias,val=0.001)
        # torch.nn.init.constant_(self.fc2.weight,val=0.001)
        # torch.nn.init.constant_(self.fc2.bias,val=0.001)
        # torch.nn.init.constant_(self.fc3.weight,val=0.001)
        # torch.nn.init.constant_(self.fc3.bias,val=0.001)
        
        
        # torch.nn.init.constant_(self.conv6.weight,val=0.01)
        # torch.nn.init.constant_(self.conv7.weight,val=0.01)
        # torch.nn.init.constant_(self.conv8.weight,val=0.01)
        # torch.nn.init.constant_(self.conv2.bias,val=0.01)

    
        
    def forward(self, x,disp=0):
         # print(x.shape)
        if(disp==1):print('-------------',x.shape)
        y = F.relu(self.conv1(x))   # 输入: (1, 96, 96), 输出: (16, 96, 96)
        y = self.pool(y)   # 输出: (16, 48, 48)
        if(disp==1):print('-------------',y.shape)
        y = F.relu(self.conv2(y))   # 输出: (32, 48, 48)
        y = self.pool(y)   # 输出: (32, 24, 24)
        if(disp==1):print('-------------',y.shape)


    
        masks = F.relu(self.conv3(y))   # 输出: (32, 32, 32)
        if(disp==1):print('-------------',masks.shape)
        
        masks = F.relu(self.conv4(masks))   # 输出: (32, 32, 32)
        if(disp==1):print('-------------',masks.shape)
        
        masks = F.relu(self.conv5(masks))   # 输出: (32, 32, 32)
        if(disp==1):print('-------------',masks.shape)
        
        
        masks = self.upsample1(masks)   # 输出: (64, 32, 32)
        if(disp==1):print('-------------',masks.shape)
        
        masks = self.upsample2(masks)   # 输出: (16, 64, 64)
        if(disp==1):print('-------------',masks.shape)
        
        
        
        
        
        y = y.view(-1, 32*y.shape[2]*y.shape[3])  # 输出: (16*5*5)
        # print(x.shape)
        y = F.relu(self.Tfc1(y)) # 输出: (120)y
        y = F.relu(self.Tfc2(y)) # 输出: (84)
        y = self.Tfc3(y) # 输出(10)

        
        #pmasks=x
        
        
        pm=torch.Tensor(x.shape[0],x.shape[2],x.shape[3]).to(device)
        pm[:,:,:]=masks[:,0,:,:]
        
        pm[pm<=0.25]=0
        pm[pm>0.25]=1
        Lxx,Lyy=pm*Lx,pm*Ly
        
        
        pm=x[:,0,:,:]*pm
        
        pmasks=torch.Tensor(x.shape[0],2,x.shape[2],x.shape[3]).to(device)
        pmasks[:,0,:,:],pmasks[:,1,:,:]=(pm+Lxx)/2,(pm+Lyy)/2

        
        # pos = F.relu(self.conv6(x))
        # pos = F.relu(self.conv7(pos))
        # pos = F.relu(self.conv8(pos))
        # self.filter(pmasks>0.9)
        
        pos = F.relu(self.pool(self.conv6(pmasks)))
        pos = F.relu(self.pool(self.conv7(pos)))
        pos = F.relu(self.pool(self.conv8(pos)))
        
        pos = F.relu(self.pool(self.conv9(pos)))
        pos = F.relu(self.pool(self.conv10(pos)))
        
        pos = pos.view(-1, 512*pos.shape[2]*pos.shape[3])  # 输出: (16*5*5)
        pos = self.Bfc1(pos)   # 输入: (1, 96, 96), 输出: (16, 96, 96)
        pos = self.Bfc2(pos) 
        pos[pos<0.1]=0
        # pos[pos>0.95]=1
        
        # pos = self.Bfc2(pos)
        # pmasks = self.pool6(pmasks)   # 输出: (16, 48, 48)
        
        # pos = pmasks.view(-1, 1*pmasks.shape[2]*pmasks.shape[3])  # 输出: (16*5*5)
        # # print(x.shape)
        # pos = F.relu(self.Bfc1(pos)) # 输出: (120)
        # pos = F.relu(self.Bfc2(pos)) # 输出: (84)
        # pos = self.Bfc3(pos) # 输出(10)

        
        return y,masks,pos



net = MLCNN().to(device)   # 用于训练的网络模型


classifier_epochs=500
classifier_alaph=1e-4
segmentation_epochs=500
segmentation_alaph=0.001
pos_epochs=1000
pos_alaph=0.0005


Batchsize=400+1#X_train.shape[0]+1

def Train_model(flag='Training_classifier'):
    global MASKS_label,masks,final_y2
    print('#######################################')
    print(flag+'------- Functions Ready----')
    print('#######################################')
    
    if(flag=='Training_classifier'):
        epochs=classifier_epochs
        alaph=classifier_alaph
        criterion = nn.BCEWithLogitsLoss()
        
    if(flag=='Training_segmentation'):
        epochs=segmentation_epochs
        alaph=segmentation_alaph
        criterion = nn.MSELoss()
        
        net.conv1.weight.requires_grad=False
        net.conv1.bias.requires_grad=False
        net.conv2.weight.requires_grad=False
        net.conv2.bias.requires_grad=False
    
        net.Tfc1.weight.requires_grad=False
        net.Tfc1.bias.requires_grad=False
        
        net.Tfc2.weight.requires_grad=False
        net.Tfc2.bias.requires_grad=False
        
        net.Tfc3.weight.requires_grad=False
        net.Tfc3.bias.requires_grad=False
        
        
        net.Bfc1.weight.requires_grad=False
        net.Bfc1.bias.requires_grad=False
        
        # net.Bfc2.weight.requires_grad=False
        # net.Bfc2.bias.requires_grad=False
        
        # net.Bfc3.weight.requires_grad=False
        # net.Bfc3.bias.requires_grad=False
        
        
    if(flag=='Training_pos_loc'):
        epochs=pos_epochs
        alaph=pos_alaph
        criterion = nn.MSELoss()#MSELoss()
    
        net.conv1.weight.requires_grad=False
        net.conv1.bias.requires_grad=False
        net.conv2.weight.requires_grad=False
        net.conv2.bias.requires_grad=False
    
        net.Tfc1.weight.requires_grad=False
        net.Tfc1.bias.requires_grad=False
        
        net.Tfc2.weight.requires_grad=False
        net.Tfc2.bias.requires_grad=False
        
        net.Tfc3.weight.requires_grad=False
        net.Tfc3.bias.requires_grad=False
        
        net.conv3.weight.requires_grad=False
        # net.conv3.bias.requires_grad=False
        net.conv4.weight.requires_grad=False
        # net.conv4.bias.requires_grad=False
        net.conv5.weight.requires_grad=False
        # net.conv5.bias.requires_grad=False
        
        net.upsample1.weight.requires_grad=False
        net.upsample2.weight.requires_grad=False
        net.upsample1.bias.requires_grad=False
        net.upsample2.bias.requires_grad=False
        
        # torch.nn.init.constant_(net.conv6.weight,val=0)
        # torch.nn.init.constant_(net.conv7.weight,val=0)
        # torch.nn.init.constant_(net.conv8.weight,val=0)
        # torch.nn.init.constant_(net.Bfc1.weight,val=0)
        # torch.nn.init.constant_(net.Bfc1.bias,val=0)
        
    #定义损失函数与优化器0
    # criterion = nn.MSELoss()#CrossEntropyLoss, MSELoss
    #criterion = nn.BCEWithLogitsLoss
    optimizer = optim.Adam(net.parameters(), lr=alaph)
    #optimizer = optim.SGD(model.parameters(), lr=alaph)
    #optimizer = optim.Adam(anom_classifier.parameters(),lr = 0.001) 
    start = time.time()
    loss_accbuf=[]
    acc1,acc2=0,0
    count=0
    
    for epoch in range(epochs):
        selnum=Y_train.shape[0]
        rands=random.sample(range(0,selnum),selnum)  
        train_loss = 0

        t1=time.time()
        for i in range(int(selnum/Batchsize)+1):
            
            rand_sels = rands[Batchsize*i:Batchsize*(i+1)]
            
            
            
            buf=X_train[rand_sels,:,:]#MASKS_train[rand_sels,0,:,:]#*
            
            # x=torch.Tensor(buf.shape[0],2,buf.shape[1],buf.shape[2]).to(device)
            # x[:,0,:,:],x[:,1,:,:]=buf*Lx,buf*Ly
            x = torch.reshape(buf,(buf.shape[0],1,buf.shape[1],buf.shape[2]))
            
            buf=Y_train[rand_sels]
            class_label = torch.reshape(buf,(buf.shape[0],1,buf.shape[1]))
            MASKS_label=MASKS_train[rand_sels,:,:,:]
            
            
            buf=pos_train[rand_sels]
            pos_label=torch.reshape(buf,(buf.shape[0],buf.shape[1])) 
            
    
            classes,masks,pos = net(x)
            classes=torch.reshape(classes,(classes.shape[0],1,classes.shape[1]))
            
            
            if(flag=='Training_classifier'):
                loss = criterion(classes,class_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                max_vals,Truths1= torch.max(class_label[:,0,:],1)
                acc1=mc.calc_accuracy(classes[:,0,:],Truths1)
            
            elif(flag=='Training_segmentation'):
                acc1,acc2=0,0
                loss = criterion(masks[:,0:1,:,:],MASKS_label[:,0:1,:,:])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            elif(flag=='Training_pos_loc'):
                # acc1,acc2=0,0
                # loss = criterion(pos[:,0:1,:,:],MASKS_label[:,1:2,:,:])
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # train_loss += loss.item()
                
                loss = criterion(pos,pos_label[:,:])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # train_loss += loss.item()
                # pred_pos=pos.cpu().detach().numpy()
                # label_pos=pos_label[:,:].cpu().detach().numpy()
                # dist1=np.linalg.norm(pred_pos- label_pos,axis=1)
                # dist1= np.sum(dist1)
            
            if(i==0): final_y,final_y1,final_y2=classes.cpu().detach().numpy(),masks.cpu().detach().numpy(),pos.cpu().detach().numpy()
            else:final_y,final_y1,final_y2=np.vstack([final_y,classes.cpu().detach().numpy()]),\
                                           np.vstack([final_y1,masks.cpu().detach().numpy()]),\
                                           np.vstack([final_y2,pos.cpu().detach().numpy()])
        
        if(flag=='Training_pos_loc'):
            all_loss = criterion(torch.from_numpy(final_y2.astype(np.float32)).to(device),pos_train[rands])#.item()
         
            train_loss=all_loss.item()
            label_pos=pos_train[rands].cpu().detach().numpy()
            dist1=np.linalg.norm(final_y2-label_pos,axis=1)
             
            dist1= np.sum(dist1)#/len(dist1)
            

            
        '----------------------------Vadiating----------------------------'   
        for i in range(int(Y_vadit.shape[0]/Batchsize)+1):
            buf=X_vadit[Batchsize*i:Batchsize*(i+1),:,:]#MASKS_vadit[Batchsize*i:Batchsize*(i+1),0,:,:]#*
            # x=torch.Tensor(buf.shape[0],2,buf.shape[1],buf.shape[2]).to(device)
            # x[:,0,:,:],x[:,1,:,:]=buf*Lx,buf*Ly
            x = torch.reshape(buf,(buf.shape[0],1,buf.shape[1],buf.shape[2]))
            classes1,masks1,pos1 = net(x)
            if(i==0): final_y,final_y1,final_y2=classes1.cpu().detach().numpy(),masks1.cpu().detach().numpy(),pos1.cpu().detach().numpy()
            else:final_y,final_y1,final_y2=np.vstack([final_y,classes1.cpu().detach().numpy()]),\
                                           np.vstack([final_y1,masks1.cpu().detach().numpy()]),\
                                           np.vstack([final_y2,pos1.cpu().detach().numpy()])


        if(flag=='Training_classifier'):
            final_y=torch.from_numpy(final_y.astype(np.float32)).to(device)
            max_vals,Truths= torch.max(Y_vadit[:,:],1)
            acc2=mc.calc_accuracy(final_y[:,:],Truths)   
            loss_accbuf.append([train_loss,acc1,acc2])
            if(train_loss<0.05):count=count+1
            if(count>10):break   
            
        
        if(flag=='Training_segmentation'):
            final_y1=torch.from_numpy(final_y1.astype(np.float32)).to(device)
            loss = criterion(final_y1[:,0:1,:,:],MASKS_vadit[:,0:1,:,:])
            vadit_loss=loss.item() 
            loss_accbuf.append([train_loss,vadit_loss])

        if(flag=='Training_pos_loc'):
            # final_y2=torch.from_numpy(final_y2.astype(np.float32)).to(device)
            # loss = criterion(final_y2[:,0:1,:,:],MASKS_vadit[:,1:2,:,:])
            # vadit_loss=loss.item() 
            # loss_accbuf.append([train_loss,vadit_loss])
            
            valdit_loss = criterion(torch.from_numpy(final_y2.astype(np.float32)).to(device),pos_vadit).item() 
            label_pos=pos_vadit.cpu().detach().numpy()
            dist2=np.linalg.norm(final_y2-label_pos,axis=1)
             
            dist2= np.sum(dist2)#/len(dist2)
            
            loss_accbuf.append([train_loss,valdit_loss,dist1,dist2])
  
  
        
        if (epoch % 10==0):
            t2=time.time()
            print('time: %0.3f'% (t2-t1))
            print('[%d, %5d]'% (epoch + 1, i + 1),loss_accbuf[-1])
            #print('[%d, %5d] loss: %0.6f  training acc %0.6f, validating acc %0.6f' % (epoch + 1, i + 1, running_loss,acc1,acc2))


    print('time = %2dm:%2ds' % ((time.time()-start)//60, (time.time()-start)%60))
    
    torch.save(net.state_dict(),'model_weights/model_cnn_nn.pth') 
    
    np.save('model_weights/'+flag+'_loss_cnn_nn_%d' % epochs+'_'+str(alaph)+'.npy',np.array(loss_accbuf)) 

net.load_state_dict(torch.load('model_weights/model_cnn_nn.pth'))

# Train_model(flag='Training_classifier')
# Train_model(flag='Training_segmentation')
# Train_model(flag='Training_pos_loc')




classifier_lossaccbf=np.load('model_weights/Training_classifier_loss_cnn_nn_%d' % classifier_epochs+'_'+str(classifier_alaph)+'.npy') 
segmentation_lossaccbf=np.load('model_weights/Training_segmentation_loss_cnn_nn_%d' % segmentation_epochs+'_'+str(segmentation_alaph)+'.npy') 
pos_lossaccbf=np.load('model_weights/Training_pos_loc_loss_cnn_nn_%d' % pos_epochs+'_'+str(pos_alaph)+'.npy') 

def dis_seg_imgs(MASKS,Pred_masks,label_pos,pred_pos,flag='',sel=6):
    plt.figure(figsize=(6,2))
    
    plt.subplots_adjust(hspace=0.1) 
    plt.subplots_adjust(wspace=0.3) 

    MASKS=MASKS.cpu().detach().numpy()
    ticks=np.round(np.linspace(0, 80,5))
    
    plt.subplot(1,3,1)
    plt.imshow(XX[sel,:,:].cpu().detach().numpy(),vmin=0,vmax=1,cmap='gray')
    plt.xticks(ticks,fontsize=8)
    plt.yticks(ticks,fontsize=8)
    
    plt.subplot(1,3,2)
    plt.imshow(MASKS[sel,0,:,:],vmin=0,vmax=1,cmap='gray')
    plt.xticks(ticks,fontsize=8)
    plt.yticks(ticks,fontsize=8)
    
    nr=int(len(label_pos[sel])/2)
    B_p1=label_pos[sel]#.reshape(nr,2)
    data=B_p1[B_p1[:]>0.01]

    print(data)
    
    n=int(len(data)/2)
    
    ms=10
    plt.scatter(data[:n]*80,data[n:2*n]*80,c='r',s=ms,marker='*')
    plt.xticks(ticks,fontsize=8)
    plt.yticks(ticks,fontsize=8)
    
    # plt.subplot(1,5,3)
    # plt.imshow(MASKS[sel,1,:,:],vmin=0,vmax=1,cmap='gray')
    # plt.xticks(ticks,fontsize=8)
    # plt.yticks(ticks,fontsize=8)
    
    plt.subplot(1,3,3)
    plt.imshow(Pred_masks[sel,0,:,:],cmap='gray')
    # plt.imshow(pred_pos[sel,0,:,:],vmin=0.5,vmax=1,cmap='gray')
    plt.xticks(ticks,fontsize=8)
    plt.yticks(ticks,fontsize=8)
    

    # B_p2=pred_pos[sel].reshape(nr,2)
    # data=B_p2[B_p2[:,0]>0.01,:]
    # data=data[data[:,1]>0.01,:]
    
    B_p2=pred_pos[sel]#.reshape(nr,2)
    data=B_p2[B_p2[:]>0.01]

    print(data)
    
    n=int(len(data)/2)
    plt.scatter(data[:n]*80,data[n:2*n]*80,c='r',s=ms,marker='*')


    
    # plt.subplot(1,5,5)
    # plt.imshow(Pred_masks[sel,1,:,:],vmin=0,vmax=1,cmap='gray')
    # plt.xticks(ticks,fontsize=8)
    # plt.yticks(ticks,fontsize=8)

    plt.savefig('saved_figs/Segment_figs/'+flag.split('_')[0]+'/'+flag+'_segmentation.png',bbox_inches='tight', dpi=300) 
    
classes_labels=['Malignant','Benign']

'--------------------Results of performance----------------------'
def get_results(flag='training_'):
    global MASKS,XX,YY,Pred_masks,class_Label,class_pred,label_pos,pred_pos,prob_true, prob_pred 

    if(flag=='training_'):
        label_MASKS=MASKS_train
        XX=X_train
        YY=Y_train
        files=train_bufs
        label_pos=pos_train[:,:].cpu().detach().numpy()
        
        
    elif(flag=='vaditing_'):
        label_MASKS=MASKS_vadit
        XX=X_vadit
        YY=Y_vadit
        files=vadit_bufs
        label_pos=pos_vadit[:,:].cpu().detach().numpy()
        
    elif(flag=='testing_'):
        label_MASKS=MASKS_test
        XX=X_test
        YY=Y_test
        files=test_bufs
        label_pos=pos_test[:,:].cpu().detach().numpy()        
        
    for i in range(int(XX.shape[0]/Batchsize)+1):
        buf=XX[Batchsize*i:Batchsize*(i+1),:,:]#MASKS_vadit[Batchsize*i:Batchsize*(i+1),1,:,:]
        x = torch.reshape(buf,(buf.shape[0],1,buf.shape[1],buf.shape[1]))
        classes,masks,pos  = net(x)
        if(i==0): 
            final_y=classes.cpu().detach().numpy()
            Pred_masks=masks.cpu().detach().numpy()
            pred_pos=pos.cpu().detach().numpy()
        else:
            final_y=np.vstack([final_y,classes.cpu().detach().numpy()])
            Pred_masks=np.vstack([Pred_masks,masks.cpu().detach().numpy()])
            pred_pos=np.vstack([pred_pos,pos.cpu().detach().numpy()])
          
    final_y=torch.from_numpy(final_y.astype(np.float32)).to(device)
    max_vals,Truths= torch.max(YY[:,:],1)
    res=mc.calc_accuracy(final_y[:,:],Truths)
    
    print(flag+'Overall Accuracy',res)
    class_Label=YY
    class_pred=final_y
    
    mmc.plot_confusion_matrix(class_pred,class_Label,sfname='CNN_'+flag,classes=['Malignant','Benign'])
    mmc.plot_loss_accuracy(classifier_lossaccbf,sfname='CNN_classifier_')
    
    
    '-------------获取专家识别结果--------------'
    # expert_preds=np.load('expert_recognition.npy')
    # expert_preds=torch.from_numpy(expert_preds.astype(np.float32)).to(device)
    # expert_truths=np.loadtxt('saved_figs/truths.txt')
    # expert_truths=torch.from_numpy(expert_truths.astype(np.float32)).to(device)
    # mmc.plot_confusion_matrix(expert_preds,expert_truths,sfname='expert_'+flag,classes=['Malignant','Benign'])

    
    mmc.plot_loss(segmentation_lossaccbf,sfname='CNN_segmentation_')
    
    
    mmc.plot_loss1(pos_lossaccbf,sfname='CNN_posloc_')
    
    mmc.plot_roc(class_Label,class_pred,['Malignant','Benign'],sfname='CNN_'+flag)  
    
    prob_true,prob_pred=mmc.plot_calibration_curve(class_pred,class_Label,sfname=flag)

    print(prob_true,prob_pred)


    #[1,5,10,35,26,67,232,344,565,896]
    
    sels=random.sample(range(0,len(YY)),10)  
    
    
    
    
    #sels=[13,45,56,232,445,567]
    #sels=[1,2,3,4,5,6]
    

    for em in sels:
        sel=em
        print(files[sel][-1])
        name=files[sel][-1].split('/')[-1]
        dis_seg_imgs(label_MASKS,Pred_masks,label_pos,pred_pos,flag=flag+name,sel=sel)

get_results(flag='training_')
