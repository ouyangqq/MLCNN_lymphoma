# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:34:15 2023
@author: Ouyangqq
"""
# import sys
import time
import numpy as np
import torch
from torch import optim 
import random
import torch.utils.data
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import metrics as mc

import Model_Metrics as mmc

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 指定GPU or CPU 进行训练

# %%
'-----Loading training and testing data for wenhou hospital-------- '
bufs=np.load("dataset/Lym_dataset.npy") 
Xd,Yd,MASK,POS=[],[],[],[]
org_imgs=[]
org_labels=[]

for em in bufs: 
    Xd.append(em[1])
    Yd.append(em[2])
    MASK.append(np.array([em[3],em[4]]))
    POS.append(np.array(em[5]))
    org_imgs.append(em[0])
    org_labels.append(em[2][1])
    
fg=int(len(bufs)/2)
seqs=np.arange(0,len(bufs),1) 

sel1=seqs[0:fg] 
sel2=seqs[fg:]   
X_train=torch.from_numpy(np.array(Xd)[sel1,:,:].astype(np.float32)).to(device)
X_vadit=torch.from_numpy(np.array(Xd)[sel2,:,:].astype(np.float32)).to(device)

Y_train=torch.from_numpy(np.array(Yd)[sel1,:].astype(np.float32)).to(device)
Y_vadit=torch.from_numpy(np.array(Yd)[sel2,:].astype(np.float32)).to(device)

MASKS_train=torch.from_numpy(np.array(MASK)[sel1,:,:,:].astype(np.float32)).to(device) 
MASKS_vadit=torch.from_numpy(np.array(MASK)[sel2,:,:,:].astype(np.float32)).to(device)

# MASKS_train[:,1,:,:]=MASKS_train[:,1,:,:]*2
# MASKS_vadit[:,1,:,:]=MASKS_vadit[:,1,:,:]*2

pos_train=torch.from_numpy(np.array(POS)[sel1,:].astype(np.float32)).to(device) 
pos_vadit=torch.from_numpy(np.array(POS)[sel2,:].astype(np.float32)).to(device)

train_bufs=bufs[:fg]
vadit_bufs=bufs[fg:]  

# %%
'---------------------Establishment of architure MLCNN model------------------------- '
class MLCNN(nn.Module): # 集成nn.Module父类
    def __init__(self):
        super(MLCNN, self).__init__()# 看一下具体的参数
        'Classfier network'
        #卷积层1：输入1个特征图，输出16个特征图，卷积核3*3，平移步长为1格，向四边扩充1格，有偏置项
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1,bias=True) 
        #卷积层1：输入16个特征图，输出32个特征图，卷积核3*3，平移步长为1格，向四边扩充1格，有偏置项
        self.conv2 = nn.Conv2d(16,32, 3,stride=1,padding=1,bias=True)
        self.pool = nn.MaxPool2d(2, 2)#池化为原来大小的1/2

       
        #全连接层1：输入数据维度12800，输入数据维度256
        self.Tfc1 = nn.Linear(12800 , 256)
        #全连接层2：输入数据维度256，输入数据维度64
        self.Tfc2 = nn.Linear(256, 2)
        #全连接层3：输入数据维度64，输入数据维度2
        
        'Segmentation network'
        #卷积层1：输入16个特征图，输出64个特征图，卷积核3*3，平移步长为1格，向四边扩充1格，无偏置项
        self.conv3 = nn.Conv2d(32,64, 3,stride=1,padding=1,bias=False)
        #卷积层1：输入32个特征图，输出64个特征图，卷积核3*3，平移步长为1格，向四边扩充1格，无偏置项
        self.conv4 = nn.Conv2d(64,128, 3,stride=1,padding=1,bias=False)
        #卷积层1：输入64个特征图，输出64个特征图，卷积核3*3，平移步长为1格，向四边扩充1格，无偏置项
        self.conv5 = nn.Conv2d(128,64, 3,stride=1,padding=1,bias=False)
        #卷积层1：输入64个特征图，输出32个特征图，卷积核3*3，平移步长为1格，向四边扩充1格，无偏置项
        self.conv6 = nn.Conv2d(64,32, 3,stride=1,padding=1,bias=False)
        
        #输入16个特征图，输出8个特征图，卷积核3*3，平移步长为1格，向四边扩充1
        self.upsample1 = nn.ConvTranspose2d(32,16,3,stride=2, padding=1, output_padding=1)
        #输入16个特征图，输出8个特征图，卷积核3*3，平移步长为1格，向四边扩充1
        self.upsample2 = nn.ConvTranspose2d(16,1, 3,stride=2, padding=1, output_padding=1)
        
        #self.amplify=F.interpolate(scale_factor=4,mode='bicubic')#最邻近插值法
        'Localization network'
        #卷积层6：输入2个特征图，输出14个特征图，卷积核3*3，平移步长1格，向四边扩充1格，无偏置项
        self.conv7 = nn.Conv2d(1,14, 3,stride=1,padding=1,bias=False) 
        #卷积层6：输入14个特征图，输出28个特征图，卷积核3*3，平移步长1格，向四边扩充1格，无偏置项
        self.conv8 = nn.Conv2d(14,28, 3,stride=1,padding=1,bias=False)
        #卷积层6：输入28个特征图，输出56个特征图，卷积核3*3，平移步长1格，向四边扩充1格，无偏置项
        self.conv9 = nn.Conv2d(28,56,3,stride=1,padding=1,bias=False)
        #卷积层6：输入56个特征图，输出224个特征图，卷积核3*3，平移步长1格，向四边扩充1格，无偏置项
        self.conv10 = nn.Conv2d(56,224,3,stride=1,padding=1,bias=False)  
        #卷积层6：输入224个特征图，输出224个特征图，卷积核3*3，平移步长1格，向四边扩充1格，无偏置项
        self.conv11 = nn.Conv2d(224,224,3,stride=1,padding=1,bias=False)
        #卷积层6：输入224个特征图，输出488个特征图，卷积核3*3，平移步长1格，向四边扩充1格，无偏置项
        self.conv12 = nn.Conv2d(224,448,3,stride=1,padding=1,bias=False)

        #全连接层1：输入数据维度128000，输入数据维度256
        self.Bfc1 = nn.Linear(448, pos_train.shape[1])

        
    def forward(self, x,disp=0):
         # print(x.shape)
        if(disp==1):print('-------------',x.shape)
        y = F.relu(self.conv1(x))   # 输入: (1, 80, 80), 输出: (16, 80, 80)
        y = self.pool(y)   # 输出: (16, 40, 40)
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
        
        masks = F.relu(self.conv6(masks))   # 输出: (32, 32, 32)
        if(disp==1):print('-------------',masks.shape)
        
        
        masks = self.upsample1(masks)   # 输出: (64, 32, 32)
        if(disp==1):print('-------------',masks.shape)
        
        masks = self.upsample2(masks)   # 输出: (16, 64, 64)
        if(disp==1):print('-------------',masks.shape)
        

        y = y.view(-1, 32*y.shape[2]*y.shape[3]) # 输出: (16*5*5)
        y = F.relu(self.Tfc1(y)) # 输出: (120)y
        y = self.Tfc2(y)
        
        pos = F.relu(self.conv7(masks))
        pos = self.pool(pos)
        pos = F.relu(self.conv8(pos))
        pos = self.pool(pos)
        pos = F.relu(self.conv9(pos))
        pos = self.pool(pos)
        pos = F.relu(self.conv10(pos))
        pos = self.pool(pos)
        pos = F.relu(self.conv11(pos))
        pos = self.pool(pos)
        pos = F.relu(self.conv12(pos))
        pos = self.pool(pos)
        
        pos = pos.view(-1, 448*pos.shape[2]*pos.shape[3])  # 输出: (448*1*1)
        pos = self.Bfc1(pos)   # 输入: (1, 96, 96), 输出: (16, 96, 96)

        
        return y,masks,pos

net = MLCNN().to(device)   # Create a MLCNN model
# %%
'-----------------------------------Training MLCNN model--------------------------------------------- '
classifier_epochs=500
classifier_alaph=1e-3
segmentation_epochs=500
segmentation_alaph=0.001
pos_epochs=500
pos_alaph=0.001


Batchsize=500+1#X_train.shape[0]+1

def Train_model(flag='Training_classifier'):
    global MASKS_label,masks,final_y2
    print('#######################################')
    print(flag+'------- Functions Ready----')
    print('#######################################')
    
    if(flag=='Training_classifier'):
        epochs=classifier_epochs
        alaph=classifier_alaph
        #定义损失函数
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
        
        net.conv3.weight.requires_grad=False
        net.conv4.weight.requires_grad=False
        net.conv5.weight.requires_grad=False
        net.conv6.weight.requires_grad=False
        
        net.upsample1.weight.requires_grad=False
        net.upsample2.weight.requires_grad=False
        net.upsample1.bias.requires_grad=False
        net.upsample2.bias.requires_grad=False
                
    #定义优化器0
    # criterion = nn.MSELoss()#CrossEntropyLoss, BCEWithLogitsLoss
    optimizer = optim.Adam(net.parameters(), lr=alaph)
    start = time.time()
    loss_accbuf=[]
    acc1,acc2=0,0    
    for epoch in range(epochs):
        selnum=Y_train.shape[0]
        rands=random.sample(range(0,selnum),selnum)  
        train_loss = 0

        t1=time.time()
        '----------------------------Training----------------------------'  
        for p in range(int(selnum/Batchsize)+1):
            rand_sels = rands[Batchsize*p:Batchsize*(p+1)]
            buf=X_train[rand_sels,:,:]#MASKS_train[rand_sels,0,:,:]#*
            x = torch.reshape(buf,(buf.shape[0],1,buf.shape[1],buf.shape[2]))
            # x =MASKS_train[rand_sels,1:2,:,:]
            classes,masks,pos = net(x)
             
            if(flag=='Training_classifier'):
                class_label=Y_train[rand_sels,:]
                loss = criterion(classes,class_label)
                # train_loss += loss.item()
            
            if(flag=='Training_segmentation'):
                MASKS_label=MASKS_train[rand_sels,0:1,:,:]+MASKS_train[rand_sels,1:2,:,:]
                loss = criterion(masks,MASKS_label)
                
            if(flag=='Training_pos_loc'):
                pos_label=pos_train[rand_sels]
                loss = criterion(pos,pos_label)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            '---------------------Combine training outputs of MLCNN for each batchsize----------------' 
            if(p==0): final_y,final_y1,final_y2=classes.cpu().detach().numpy(),\
                                                masks.cpu().detach().numpy(),\
                                                pos.cpu().detach().numpy()
                                                
            else:final_y,final_y1,final_y2=np.vstack([final_y,classes.cpu().detach().numpy()]),\
                                            np.vstack([final_y1,masks.cpu().detach().numpy()]),\
                                            np.vstack([final_y2,pos.cpu().detach().numpy()])
            '-------------------------------------------------------------------------------'                    

    

        '----------------------------Get training results for each epoch----------------------------'  
        final_y,final_y1,final_y2=torch.from_numpy(final_y.astype(np.float32)).to(device),\
                                  torch.from_numpy(final_y1.astype(np.float32)).to(device),\
                                  torch.from_numpy(final_y2.astype(np.float32)).to(device)
        
        if(flag=='Training_classifier'):
            all_loss = criterion(final_y,Y_train[rands,:])
            train_loss=all_loss.item()
            max_vals,Truths1= torch.max(Y_train[rands,:],1)
            acc1=mc.calc_accuracy(final_y,Truths1)
            
        if(flag=='Training_segmentation'):  
            all_loss = criterion(final_y1[:,:,:,:],MASKS_train[rands,0:1,:,:]+MASKS_train[rands,1:2,:,:])
            train_loss=all_loss.item()
            
        if(flag=='Training_pos_loc'):
            all_loss = criterion(final_y2,pos_train[rands])
            train_loss=all_loss.item()
            label_pos=pos_train[rands].cpu().detach().numpy()
            dist1=np.linalg.norm(final_y2.cpu().detach().numpy()-label_pos,axis=1)
            dist1= np.sum(dist1)#/len(dist1)
            
        
        '----------------------------Get vaditing results for each epoch----------------------------'   
        for i in range(int(Y_vadit.shape[0]/Batchsize)+1):
            buf=X_vadit[Batchsize*i:Batchsize*(i+1),:,:]#MASKS_vadit[Batchsize*i:Batchsize*(i+1),0,:,:]#*
            x = torch.reshape(buf,(buf.shape[0],1,buf.shape[1],buf.shape[2]))
            classes1,masks1,pos1 = net(x)
            '---------------------Combine valditing outputs of MLCNN for each batchsize----------------'
            if(i==0): final_y,final_y1,final_y2=classes1.cpu().detach().numpy(),masks1.cpu().detach().numpy(),pos1.cpu().detach().numpy()
            else:final_y,final_y1,final_y2=np.vstack([final_y,classes1.cpu().detach().numpy()]),\
                                            np.vstack([final_y1,masks1.cpu().detach().numpy()]),\
                                            np.vstack([final_y2,pos1.cpu().detach().numpy()])

        final_y,final_y1,final_y2=torch.from_numpy(final_y.astype(np.float32)).to(device),\
                                  torch.from_numpy(final_y1.astype(np.float32)).to(device),\
                                  torch.from_numpy(final_y2.astype(np.float32)).to(device)

        if(flag=='Training_classifier'):
            max_vals,Truths= torch.max(Y_vadit[:,:],1)
            acc2=mc.calc_accuracy(final_y[:,:],Truths)   
            loss_accbuf.append([train_loss,acc1,acc2])

        if(flag=='Training_segmentation'):
            MASKS_label=MASKS_vadit[:,0:1,:,:]+MASKS_vadit[:,1:2,:,:]
            loss = criterion(final_y1,MASKS_label)
            vadit_loss=loss.item() 
            loss_accbuf.append([train_loss,vadit_loss])

        if(flag=='Training_pos_loc'):            
            valdit_loss = criterion(final_y2,pos_vadit).item() 
            label_pos=pos_vadit.cpu().detach().numpy()
            dist2=np.linalg.norm(final_y2.cpu().detach().numpy()-label_pos,axis=1)
            dist2= np.sum(dist2)
            loss_accbuf.append([train_loss,valdit_loss,dist1,dist2])
  
        if (epoch % 10==0):
            t2=time.time()
            print('time: %0.3f'% (t2-t1))
            print('[%d, %5d]'% (epoch + 1, i + 1),loss_accbuf[-1])

    print('time = %2dm:%2ds' % ((time.time()-start)//60, (time.time()-start)%60))
    torch.save(net.state_dict(),'model_weights/model_cnn_nn.pth') 
    np.save('model_weights/'+flag+'_loss_cnn_nn_%d' % epochs+'_'+str(alaph)+'.npy',np.array(loss_accbuf)) 


# Train_model(flag='Training_classifier')
# Train_model(flag='Training_segmentation')
# Train_model(flag='Training_pos_loc')

# %% 
net.load_state_dict(torch.load('model_weights/model_cnn_nn.pth')) 
'-----------------------------------Evaluation of MLCNN model--------------------------------------------- '  
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
    
    B_p1=label_pos[sel]#.reshape(nr,2)
    data=B_p1[B_p1>0.01]

    print('---labeled pos------',data)
    
    n=int(len(data)/2)
    
    ms=10
    plt.scatter(data[:n]*80,data[n:2*n]*80,c='r',s=ms,marker='*')
    plt.xticks(ticks,fontsize=8)
    plt.yticks(ticks,fontsize=8)
    
    
    plt.subplot(1,3,3)
    # plt.imshow(Pred_masks[sel,0,:,:],cmap='gray')
    img=Pred_masks[sel,0,:,:]
    img=(img-img.min())/(img.max()-img.min())
    plt.imshow(img,vmin=0.2,vmax=0.5,cmap='gray')
    plt.xticks(ticks,fontsize=8)
    plt.yticks(ticks,fontsize=8)
    
    B_p2=pred_pos[sel]#.reshape(nr,2)
    data=B_p2[(B_p2>0.05)&(B_p2<0.95)]

    print('---predict pos------',data)
    
    n=int(len(data)/2)
    plt.scatter(data[:n]*80,data[n:2*n]*80,c='r',s=ms,marker='*')

    plt.savefig('saved_figs/Segment_figs/'+flag.split('_')[0]+'/'+str(sel)+'_'+flag+'_segmentation.png',bbox_inches='tight', dpi=300) 
    
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

    mmc.plot_loss(segmentation_lossaccbf,sfname='CNN_segmentation_')
    
    mmc.plot_loss1(pos_lossaccbf,sfname='CNN_posloc_')
    
    mmc.plot_roc(class_Label,class_pred,['Malignant','Benign'],sfname='CNN_'+flag)  
    
    prob_true,prob_pred=mmc.plot_calibration_curve(class_pred,class_Label,sfname=flag)

    print(prob_true,prob_pred)
    
    sels=random.sample(range(0,len(YY)),200)  
    
    for em in [2]:
        sel=em
        print(sel,files[sel][-1])
        name=files[sel][-1].split('/')[-1]
        dis_seg_imgs(label_MASKS,Pred_masks,label_pos,pred_pos,flag=flag+name,sel=sel)

get_results(flag='training_')
#%%