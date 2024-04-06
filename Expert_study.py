# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:13:40 2024

@author: qiang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:34:15 2023

@author: asus
"""

import numpy as np
import torch
import torch.utils.data
import os
import Model_Metrics as mmc






res_buf=[]
fpaths=os.listdir(r"saved_figs/Tested_orgimgs1(恶性标0,良性标1)/")
for i,pn in enumerate(fpaths):#[23:28]
    res=int(pn.split('_')[1])
    oos=np.zeros(2)
    if(res==0):oos[0]=1
    elif(res==1):oos[1]=1
    
    res_buf.append(oos)

np.save('model_weights/expert_recognition.npy',np.array(res_buf))

Truths=np.loadtxt('saved_figs/truths.txt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 指定GPU or CPU 进行训练


'-------------获取专家识别结果--------------'
expert_preds=np.load('model_weights/expert_recognition.npy')
expert_preds=torch.from_numpy(expert_preds.astype(np.float32)).to(device)
expert_truths=np.loadtxt('saved_figs/truths.txt')
expert_truths=torch.from_numpy(expert_truths.astype(np.float32)).to(device)
mmc.plot_confusion_matrix(expert_preds,expert_truths,sfname='expert_',classes=['Malignant','Benign'])


