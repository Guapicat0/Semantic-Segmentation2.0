import os
import random

 
segfilepath=r"./pile/JPEGImages/"
saveBasePath=r"./pile/ImageSets/Segmentation/"
 
trainval_percent=0.9      #训练集0.8 测试集0.1 验证集0.1
# 9:1
train_percent=0.8
val_percent=0.1
test_percent=0.1
temp_seg = os.listdir(segfilepath)


total_seg = []
for seg in temp_seg:
    if seg.endswith(".jpg"):
        total_seg.append(seg)

num=len(total_seg)  #图片总个数
list=range(num)  
tv=int(num * trainval_percent)   #训练块的个数tv=900*0.9
te=int(num-tv)                   #测试块的个数te

tr=int(tv * train_percent)

trainval= random.sample(list, tv)
test =set(list)-set(trainval)       #set(a)-set(b）为求a与b的差集
train=random.sample(trainval,tr)
val = set(trainval)-set(train)
print("训练块有",len(trainval)," 训练集有",len(train)," 测试集有",len(test)," 验证集有",len(val))


#print(total_seg[2][:-4])    #对于abcd.png，total_seg[i][:-4]只记录了abcd

print("train and val size",tv)
print("traub size",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i  in list:  
    name=total_seg[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        if i in val:
            fval.write(name)  
    if i in test:
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
