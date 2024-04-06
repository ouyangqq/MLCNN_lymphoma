# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 09:02:36 2023

@author: asus
"""

import json
from PIL import Image, ImageDraw
import numpy as np
#import cv2
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

def gen_circle_points(center_x,center_y,r1,r2,num_points):
    theta = np.linspace(0, 2*np.pi, num_points)
    ra = r1 * np.ones((num_points))
    rb = r2 * np.ones((num_points))
    coordinates = np.column_stack([ra * np.cos(theta), rb * np.sin(theta)])
    coordinates=coordinates+np.array([center_x,center_y])
    
    return coordinates

def make_labels(image_dir, save_dir,bufs,showfig=False):
    global res
    global st
    global end
    global cut_img
    global T_mask
    global B_mask
    
    data = os.listdir(image_dir)
    temp_data = []
    for i in data:
        if i.split('.')[1] == 'json':
            temp_data.append(i)
        else:
            continue
    
    for js in temp_data:
        # print(js)
        json_data = json.load(open(os.path.join(image_dir, js), 'rb'))
        shapes_ = json_data['shapes']  # 得到标签
        
        orgin=Image.open(os.path.join(image_dir, js.replace('json', 'jpg'))).convert("L")  #转化为2D图片的结果 
        
        orgin_img=np.array(orgin) #AAAAA
    
        label = shapes_[-1]['label'] #获得当前mask的标签
        if(label=='jt'):
            points = shapes_[-1]['points'] #获得当前mask标记点坐标
            #print("wsy----",points)
            
            points=np.float32(points)
            
            st=points[points[:,0]==min(points[:,0]),:][0]
            
            end=points[points[:,0]==max(points[:,0]),:][0]
            
            st=np.int32(st)
            end=np.int32(end)
            
            #print("wsy----",st,end)
            
            cut_img=orgin_img[st[1]:end[1],st[0]:end[0]]/orgin_img[st[1]:end[1],st[0]:end[0]].max()
        else:
            print(f"{image_dir+'/'+js} 没有标截图")
            sys.exit()
    
        cut_img=np.array(cut_img) #EEEEE
        if(cut_img.shape[0]<=10):
            print(f"{image_dir+'/'+js} 读取错误")
            sys.exit()
             

        if(image_dir.split("/")[0]==path1[0:-1]):
            oos=[0,1]
        
        if(image_dir.split("/")[0]==path2[0:-1]):
            oos=[1,0]
            
        class_label=np.array(oos) #BBBBBB
        
        T_mask=np.array(Image.new('P',cut_img.T.shape ))
        B_mask=np.array(Image.new('P',cut_img.T.shape ))
        
        if(len(shapes_[0:-1])%2!=0):
                print(f"{image_dir+'/'+js} T/B少标记")
                sys.exit()
            
        iml=80
        B_poss=np.zeros([7,2])
        count=0
        
        tpbuf=[]
        bpbuf=[]
        bpbuf0=[]
        
        selected_shapes=shapes_[0:-1]
        if(len(shapes_[0:-1])>14):selected_shapes=shapes_[0:14]
        
        for m, shape_ in enumerate(selected_shapes):  # 由于一张图片可能有多个标签，所以要遍历一遍shape #m是从“0”开始
            #Image.open(os.path.join(image_dir, js.replace('json', 'jpg'))).size
            mask = Image.new('P', cut_img.T.shape)
            
            
            label = shape_['label'] #获得当前mask的标签

            

            
            if(m%2==0)&(label[0]=='T'):
                
                tumor_points = shape_['points'] #获得当前mask标记点坐标
                tpbuf.append(tumor_points)
                points=tumor_points-st
                points = tuple(tuple(i) for i in points) #对mask标记点坐标进行整合形成一个区域
                
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.polygon(points, fill=1)  # 对图片进行画多边形的操作
                # plt.subplot(1, len(shapes_)+1, m+3) #m+3表示第m的病例均有2个标签，分别为 T_mask B_mask m从0开始
                mask_img=np.array(mask) 
                
                
                T_mask=T_mask+mask_img #CCCCCC
                
                
            elif(m%2==1)&(label[0]=='B'):
                
                biopsy_points = shape_['points'] #获得当前mask标记点坐标
                
                bpbuf0.append(biopsy_points)
                bpbuf.append(np.average(biopsy_points,0))
                # print("******",bpbuf[-1])
                
                points=biopsy_points-st
                points = tuple(tuple(i) for i in points) #对mask标记点坐标进行整合形成一个区域
                
                B_pos=np.average(points,0)

                
                r1,r2=3*cut_img.shape[1]/iml,3*cut_img.shape[0]/iml
                c_points=gen_circle_points(B_pos[0],B_pos[1],r1,r2,15)
                c_points = tuple(tuple(m) for m in c_points)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.polygon(c_points, fill=1)  # 对图片进行画多边形的操作
                
                mask_img=np.array(mask) 
                
                if(B_pos[1]>mask_img.shape[0]):
                    print(B_pos[1],mask_img.shape)
                    print(points)
                    print(f"{image_dir+'/'+js} 顺序错误或T/B标错")
                    sys.exit()
                
                # loc=np.int(B_pos)
                # mask_img[loc[0],loc[1]]=1
                B_mask=B_mask+mask_img #DDDDDD
                
                
                B_pos[0]=B_pos[0]/mask_img.shape[1]#*80
                B_pos[1]=B_pos[1]/mask_img.shape[0]#*80
                
                
                B_poss[count,0],B_poss[count,1]=B_pos[0],B_pos[1]
                
                 
                count=count+1
            else: 
                print(f"{image_dir+'/'+js} 顺序错误或T/B标错")
                sys.exit()
            
            # plt.imshow(mask_img, vmin=0, vmax=1,aspect='auto')
            #mask.save( os.path.join(save_dir, 'mask_'+label+js.replace('json', 'png')) )
            #plt.imsave(os.path.join(save_dir, 'mask_'+label+js.replace('json', 'png')) ,mask_img)

        # Image.NEAREST ：低质量, Image.BILINEAR：双线性, Image.BICUBIC ：三次样条插值,  Image.ANTIALIAS：高质量
        cut_img=Image.fromarray(cut_img)             
        cut_img=cut_img.resize((iml,iml))  
        
        T_mask=Image.fromarray(T_mask)             
        T_mask=T_mask.resize((iml,iml)) 
        
        B_mask=Image.fromarray(B_mask)             
        B_mask=B_mask.resize((iml,iml)) 
        #print(B_poss)
        Tmp=np.float32(B_mask)
        cut_img,T_mask,B_mask=np.array(cut_img),np.array(T_mask),Tmp

        if(B_mask.max()==0): 
            print(f"{image_dir+'/'+js} mask 错误")
            sys.exit()
            
        if(showfig[0]==True):

            
            if(showfig[1]=='one'):
                plt.figure(figsize=(2.2, 2.5))#宽度为9长度为3
                plt.subplot(1, 1, 1) #生成子图的函数 1行 n个mask图片均对应1个原始图片，一共n+1个图片 原始图片排在最左侧的第1位
                plt.imshow(orgin_img,aspect='auto',cmap='gray') #转化为灰度图片，将灰度图片显示的矩阵结构转化为符点结构
                for em in tpbuf:
                    tumor_points=np.vstack([np.array(em),np.array(em)[-1,:]])
                    plt.plot(tumor_points[:,0],tumor_points[:,1],'g-')
                plt.scatter(np.array(bpbuf)[:,0],np.array(bpbuf)[:,1],c='r',s=20,marker='o')
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                
                comb_str=save_dir.split('/')[0].split('\\')[1]
            
            
            if(showfig[1]=='all'):
                plt.figure(figsize=(10, 1.6))#宽度为9长度为3
                plt.subplots_adjust(hspace=0.1) 
                plt.subplots_adjust(wspace=0.4) 
                plt.subplot(1, 4, 1) #生成子图的函数 1行 n个mask图片均对应1个原始图片，一共n+1个图片 原始图片排在最左侧的第1位
                plt.imshow(orgin_img,aspect='auto',cmap='gray') #转化为灰度图片，将灰度图片显示的矩阵结构转化为符点结构
                for em in tpbuf:
                    tpoints=np.vstack([np.array(em),np.array(em)[0,:]])
                    plt.plot(tpoints[:,0],tpoints[:,1],'g-',linewidth=1)
                #plt.scatter(np.array(bpbuf)[:,0],np.array(bpbuf)[:,1],c='r',s=20,marker='o')
                
                for em in bpbuf0:
                    bpoints=np.vstack([np.array(em),np.array(em)[0,:]])
                    plt.plot(bpoints[:,0],bpoints[:,1],'r-',linewidth=1)
                
                ticks=np.round(np.linspace(0, iml,5))
                st
                end
                
                #(left,bottom),width,height,
                rect=mpatches.Rectangle((st[0],st[1]),end[0]-st[0],end[1]-st[1], 
                                        fill=False,color="yellow",linewidth=1)
                       #facecolor="red")
                plt.gca().add_patch(rect)

                
                
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                
                plt.subplot(1, 4, 2) #生成子图的函数 1行 n个mask图片均对应1个原始图片，一共n+1个图片 原始图片排在最左侧的第2位
                plt.imshow(cut_img, vmin=0, vmax=1,aspect='auto',cmap='gray') #转化为灰度图片，将灰度图片显示的矩阵结构转化为符点结构
                plt.xticks(ticks,fontsize=8)
                plt.yticks(ticks,fontsize=8)
                
                
                plt.subplot(1, 4, 3) #m+3表示第m的病例均有2个标签，分别为 T_mask B_mask m从0开始
                plt.imshow(T_mask, vmin=0, vmax=1,aspect='auto',cmap='gray')
                plt.xticks(ticks,fontsize=8)
                plt.yticks(ticks,fontsize=8)
                
                ax=plt.subplot(1, 4, 4) #m+3表示第m的病例均有2个标签，分别为 T_mask B_mask m从0开始
                ax.imshow(B_mask, vmin=0, vmax=1,aspect='auto',cmap='gray')
                
                
                data=B_poss[B_poss[:,0]>0,:]
                # print('ooooooooooooooooooo',data)
                ax.scatter(data[:,0]*iml,data[:,1]*iml,c='r',s=6,marker='*')
                plt.xticks(ticks,fontsize=8)
                plt.yticks(ticks,fontsize=8)
            
            comb_str=save_dir.split('/')[0].split('\\')[1]
            plt.savefig('saved_figs/preprocessing_imgs/'+'/'+showfig[1]+'/'+comb_str+"_"+js.replace('json', 'png'),bbox_inches='tight', dpi=300)
            
        bufs.append([orgin_img,cut_img,class_label,T_mask,B_mask,B_poss.reshape(14),image_dir+'/'+js])

def vis_label(img):
    img = Image.open(img)
    img = np.array(img)
    print(set(img.reshape(-1).tolist()))
    
    
def gen_gaussian_noise(signal,SNR):
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise=np.random.randn(*signal.shape) # *signal.shape 获取样本序列的尺寸
    noise=noise-np.mean(noise)
    signal_power=(1/signal.shape[0])*np.sum(np.power(signal,2))
    noise_variance=signal_power/np.power(10,(SNR/10))
    noise=(np.sqrt(noise_variance)/np.std(noise))*noise
    return noise




bufs=[]

path1 = r"0\0_labelmeSegmentaion/"
path2 = r"1\1_labelmeSegmentaion/"
'''
for path in [path1,path2]:
    fpaths=os.listdir(path)
    for i,pn in enumerate(fpaths[23:28]):#[23:28]
        print(i,pn)
        make_labels(path+pn, path+pn,bufs,showfig=[False,'all']) # 第一个Path+pn代表读取原始图片的路径，第二个代表保存mask和合并图的路径
    
path1 = r"0\0_InternalValidation_readbyUlTRASOUNDGRAPHERorAI/"
path2 = r"1\1_InternalValidation_readbyUlTRASOUNDGRAPHERorAI/"
for path in [path1,path2]:
    fpaths=os.listdir(path)
    for i,pn in enumerate(fpaths[23:28]):
        print(i,pn)
        make_labels(path+pn, path+pn,bufs,showfig=[False,'all']) # 第一个Path+pn代表读取原始图片的路径，第二个代表保存mask和合并图的路径

np.save("Lym_dataset_no_enhance.npy",bufs)

bufs1=[]

path1 = r"0\0_internationaltesting/"
path2 = r"1\1_internationaltesting/"
for path in [path1,path2]:
    fpaths=os.listdir(path)
    for i,pn in enumerate(fpaths):
        print(i,pn)
        make_labels(path+pn, path+pn,bufs1,showfig=[False,'all']) # 第一个Path+pn代表读取原始图片的路径，第二个代表保存mask和合并图的路径


np.save("Lym_dataset_international_test.npy",bufs1)
'''

def generate_rands_data(bufs=np.load("Lym_dataset_no_enhance.npy"),fn='',enhance=0):
    bufss1=[]
    for i in range(enhance*2+1):
        for j in range(len(bufs)) :
            em=bufs[j].copy()
            tmp=em[5].copy()
            if(i==1):em[1]=bufs[j][1]+gen_gaussian_noise(bufs[j][1],30)
            if(i==2):
                em[0]=np.flip(bufs[j][0],axis=1)
                em[1]=np.flip(bufs[j][1],axis=1)
                em[3]=np.flip(bufs[j][3],axis=1)
                em[4]=np.flip(bufs[j][4],axis=1)
                for m,dt in enumerate(tmp[0::2]): 
                    if(dt>0.01):tmp[2*m]=1-dt
                    
            '---------对biopsy位置按照x坐标从小到大排序--------'
            tmp2=tmp.reshape(7,2)
            tmp1=np.sort(tmp2[tmp2[:,0]>0.01,0]) 
            tmp3=np.zeros([7,2])
            for m,dt in enumerate(tmp1):tmp3[m,:]=tmp2[tmp2[:,0]==dt,:]
            tmp=tmp3.reshape(14)
            tmpp=np.zeros(14)
            mm=tmp[0::2][tmp[0::2]>0]
            mm1=tmp[1::2][tmp[1::2]>0]
            
            tmpp[0:len(mm)]=mm
            tmpp[len(mm):len(mm)+len(mm1)]=mm1
            em[5]=tmpp
            '-------------------------------------------'
            bufss1.append(em)
       
    
         
    sel=32
    print('***',bufss1[sel][5]) 
    print('###',bufss1[len(bufs)*enhance+sel][5])
       
    import random
    rnds=np.array(random.sample(range(0,len(bufss1)),len(bufss1)))
    
    bufss1=np.array(bufss1)[rnds,:]
    
    # POS=[]
    # for em in bufs: POS.append(np.array(em[5]))   
    # POS=np.array(POS)
     
    np.save(fn+"Lym_dataset.npy",bufss1)
    return bufss1

# Abufss=generate_rands_data(enhance=1)

Abufss=generate_rands_data(bufs=np.load("Lym_dataset_international_test.npy"),fn='test',enhance=0)



