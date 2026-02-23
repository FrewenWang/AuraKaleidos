import torch
from torch.utils.data import DataLoader
#这个模型在使用之前需要先运行一下15_2_3生成作者定制的人脸图片数据集
#数据部分只留下几个样例，以方便读者顺利运行调试代码
from _15_2_2 import *
from _15_2_4 import *
device = "cuda"
net=SiameseNetwork().to(device)

criterion=ContrastiveLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.001)

counter=[]
loss_history=[]
iteration_number=0

batch_size = 2
path_file = "./dataset/lfw-path_file.txt"
train_dataset = MyDataset(path_file=path_file)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)

for epoch in range(0,20):
    for i,data in enumerate(train_loader,0):
        img0,img1,label=data
        img0,img1,label=img0.float().to(device),img1.float().to(device),label.to(device)
        optimizer.zero_grad()
        output1,output2=net(img0,img1)

        loss_contrastive=criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()

        if i % 2 ==0:
            print('epoch:{},loss:{}\n'.format(epoch,loss_contrastive.item()))
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
