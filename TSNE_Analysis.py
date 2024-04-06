# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:10:01 2022

@author: qiang
"""

## importing the required packages
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection)
## Loading and curating the data
import metrics as mc



bufs=np.load("dataset/Lym_dataset.npy") 

Xd=[]
Yd=[]
MASK=[]
POS=[]
for em in bufs: 
    Xd.append(em[1])
    Yd.append(em[2])
    MASK.append(np.array([em[3],em[4]]))
    POS.append(np.array(em[5]))
 

seqs=np.arange(0,len(bufs),1) 
fg=int(len(bufs)/2)
sel1=seqs[0:fg] 
sel2=seqs[fg:]   
X_train=np.array(Xd)[sel1,:,:]
X_vadit=np.array(Xd)[sel2,:,:]

Y_train=np.array(Yd)[sel1,:]
Y_vadit=np.array(Yd)[sel2,:]





savflag="Valid"#"Valid"



if(savflag=="Train"):
    X =X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    y=np.log2((2*Y_train[:,-1]+Y_train[:,-2]))#4*Train[:,-1]+
elif(savflag=="Valid"):
    X =X_train.reshape(X_vadit.shape[0],X_vadit.shape[1]*X_vadit.shape[2])
    y=np.log2((2*Y_vadit[:,-1]+Y_vadit[:,-2]))#4*Train[:,-1]+
# y=Train[:,-1]      
# y=np.log2((8*Train[:,-1]+4*Train[:,-2]+2*Train[:,-3]+Train[:,-4]))

tlabels=mc.classes


n_samples, n_features = X.shape
n_neighbors = 30



# Computing t-SNE

# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=3)
# t0 = time()
# X_tsne = tsne.fit_transform(X)
# np.save("gdata/"+savflag+"_lnm_tsne.npy",X_tsne)
# print('time consume:',time() - t0)


X_tsne=np.load("model_weights/"+savflag+"_lnm_tsne.npy")

#plot_embedding(X_tsne,"t-SNE embedding of the digits (time %.2fs)" %(time() - t0))

ax = plt.figure(figsize=(3,3))

ax=plt.subplot(1,1,1)  
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')
ax.spines['bottom'].set_linewidth(1); ##设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);   ##设置左边坐标轴的粗细

lgs=np.arange(len(set(y)))


for m,c,i,target_name in zip(mc.markers,mc.colors,lgs,tlabels):
    print(target_name)
    plt.scatter(X_tsne[y==i,0], X_tsne[y==i,1],c='w',edgecolors=c,marker=m,label=target_name,s=5)

x1,x2=-60,60
plt.xticks(np.arange(x1,x2+20,20),fontsize=10,rotation=0)# ,fontweight='bold'
plt.yticks(np.arange(x1,x2+20,20),fontsize=10,rotation=0)
plt.xlim(x1,x2)
plt.ylim(x1,x2)


# if(savflag=="Valid"):plt.legend(bbox_to_anchor=(1.5, 0.99),loc=1,ncol=1,fontsize=10)
plt.legend(loc='upper right',ncol=1)# bbox_to_anchor=(0.35, 0.99)


plt.savefig('saved_figs/tsne_first_for_'+savflag+'.png',bbox_inches='tight', dpi=300)
plt.show()