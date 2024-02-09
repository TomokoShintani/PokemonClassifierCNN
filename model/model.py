from load_data import train_dataset, valid_dataset
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import torch.nn as nn

#-----訓練検証用のDataLoader作成-----#
batch_size = 4

#dataloader作る
dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

#ここで注意したいのは、この時点では画像データの次元が(RGB, H, W)になってるので
#訓練時にfor文で画像データ取得した際にちゃんと(H, W, RGB)に直さないいといけない


#-----モデルの定義-----#
model_1 = nn.Sequential(nn.Conv2d(3, 32, 3),                #(120, 120, 3) --> (118, 118, 32)
                        nn.MaxPool2d(2),                    #(118, 118, 32) --> (59, 59, 32)
                        nn.ReLU(),
                        nn.Dropout(0.25, inplace=False),    #ここをinplace=Falseにしないとloss.backward()がうまくいかない。
                        
                        nn.Conv2d(32, 64, 3),               #(59, 59, 32) --> (57, 57, 64)
                        nn.MaxPool2d(2),                    #(57, 57, 64) --> (28, 28, 64)
                        nn.ReLU(),
                        nn.Dropout(0.25, inplace=False),    
                        
                        nn.Conv2d(64, 128, 3),              #(28, 28, 64) --> (26, 26, 128)
                        nn.MaxPool2d(2),                    #(26, 26, 128) --> (13, 13, 128)
                        nn.ReLU(),
                        nn.Dropout(0.25, inplace=False),
                        
                        nn.Flatten(),                       #(13*13*128)
                        nn.Linear(13*13*128, 64),           #(13*13*128) --> (64)
                        nn.ReLU(),
                        nn.Linear(64, 3)                    #(64) --> (3)
)

# Heの初期化
def init_weights(model):
    if type(model) == nn.Linear or type(model) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(model.weight)
        model.bias.data.fill_(0.0)


#-----学習-----#
n_epochs = 10
lr = 0.0003
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=lr)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#パラメータの初期化
model_1.apply(init_weights)
model_1.to(device)

x_stock = []
t_type_stock = []
pred_stock = []

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    
    model_1.train()
    n_train = 0
    acc_train = 0
    
    for t_type, x in dataloader_train:
        n_train += t_type.size()[0]
        
        model_1.zero_grad() #勾配の初期化
        
        x = x.float()
        x = x.to(device)
        
        t_type = t_type.to(torch.int64)
        t_type = t_type.to(device)
        
        y = model_1.forward(x) #順伝播
        #print("順伝播後のy: {}".format(y))
        
        loss = loss_func(y, t_type)
        
        loss.backward() #誤差の逆伝播
        
        optimizer.step() #パラメータの更新
        
        pred = y.argmax(dim = 1)
        #print("訓練時のpred: {}".format(pred))
        
        acc_train += (pred==t_type).float().sum().item()
        
        losses_train.append(loss.tolist())
        
        
    model_1.eval()
    n_val = 0
    acc_val = 0
    
    with torch.no_grad():
        for t_type, x in dataloader_valid:
            n_val += t_type.size()[0]
            
            x_stock.extend(x)
            t_type_stock.extend(t_type)
            
            x = x.float()
            x = x.to(device)
            
            t_type = t_type.to(torch.int64)
            t_type = t_type.to(device)
            
            
            y = model_1.forward(x)
            
            loss = loss_func(y, t_type)
            
            pred = y.argmax(dim=1)
            pred_stock.extend(pred)
            
            acc_val += (pred == t_type).float().sum().item()
            
            losses_valid.append(loss.tolist())
            

    
    print("EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Validation [Loss: {:.3f}, Accuracy: {:.3f}]".format(
        epoch,
        np.mean(losses_train),
        acc_train/n_train,
        np.mean(losses_valid),
        acc_val/n_val))


#-----モデルが間違ったラベルを予測した画像をラベルとともに見てみる-----#
    
img_lst = []
label_lst = []
dict_1 = {'0':'Grass', '1':'Fire', '2':'Water'}


for img, teacher, prd in zip(x_stock, t_type_stock, pred_stock):
    if teacher != prd:
        img_lst.append(img)
        label_lst.append(str(prd.item()))
        
label_lst = list(map(lambda key: dict_1.get(key), label_lst))

fig = plt.figure(figsize = (12, 20))
fig.subplots_adjust(hspace=0.5)

for i in range(81):
    plt.subplot(9, 9, i+1)
    img = img_lst[i]
    label = label_lst[i]
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.title(label, fontsize=8)
plt.show()