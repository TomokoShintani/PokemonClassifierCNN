import os
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

table_data = pd.read_csv('../dataset/pokemon.csv')
image_directory = '../dataset/images/images/'

#-----データセットの作成-----#

#画像データにラベルを付ける
image_df=[]

for filename in os.listdir(image_directory):
    if filename.endswith(".png"):
        image_name = filename.split('.')[0]

        image = cv2.imread(os.path.join(image_directory, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #np.array(image) : (H,W,RGB)

        #画像の名前と画像のデータがセットになったdictionaryを作成
        image_df.append({'Name':image_name, 'image':np.array(image)})
        
image_dataframe = pd.DataFrame(image_df)
        
#ポケモンの名前とTypeが入ってるtable_dataに画像データをくっつける
data = table_data.merge(image_dataframe, how = 'inner', on = 'Name')


#-----特徴量エンジニアリング-----#

# Grass, Fire, Waterの三値分類タスクにする
data = data.loc[data['Type1'].isin(['Grass', 'Fire', 'Water'])]

mapping = {'Grass':0, 'Fire':1, 'Water': 2}
data['Type1'] = data['Type1'].map(mapping)

## Augmentation

# 計算高速化のため、arrayのarrayにする
image_array = np.array([np.array(img) for img in data['image']])
# Pytorchのモジュールの仕様上(..., H, W)の順番にする
image = np.transpose(torch.tensor(np.array(image_array.astype(np.int32))), (0, 3, 1, 2))

# Augmentation無し
dataset = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']), image/255.0)

# Random Affine (上下左右平行移動)
transform_shift_1 = transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))])

dataset_shift_1 = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']), transform_shift_1(image/255.0))

transform_shift_2 = transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.4, 0.4))])

dataset_shift_2 = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']), transform_shift_2(image/255.0))

# Center Crop (拡大)
transform_zoom = transforms.Compose([transforms.CenterCrop(size=100),transforms.Resize(120)])

dataset_zoom = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']), transform_zoom(image/255.0))

# Random Horizontal Flip
transform_flip = transforms.Compose([transforms.RandomHorizontalFlip(p = 1.0)])

dataset_flip = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']), transform_flip(image/255.0))

# Flip Shift
transform_flip_shift = transforms.Compose([transforms.RandomHorizontalFlip(p = 0.7), transforms.RandomAffine(degrees=0, translate=(0.3, 0.3))])

dataset_flip_shift = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']), transform_flip_shift(image/255.0))

# Random Rotation
transform_rotate= transforms.Compose([transforms.RandomRotation(degrees = (30, 330))])

dataset_rotate= torch.utils.data.TensorDataset(torch.Tensor(data['Type1']), transform_rotate(image/255.0))

# Rotate Shift
transform_rotate_shift = transforms.Compose([transforms.RandomRotation(degrees = (30, 330)), transforms.RandomAffine(degrees=0, translate=(0.3, 0.3))])

dataset_rotate_shift = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']), transform_rotate_shift(image/255.0))

# 画像表示
fig = plt.figure(num='Rotate Shift', figsize=(12, 20))
#fig.subplots_adjust(hspace=0.5)

dataloader_rotate_shift = torch.utils.data.DataLoader(dataset_rotate_shift, batch_size=1, shuffle=True)


def show_rotate_shift():
    i = 0
    for label, x in dataloader_rotate_shift:
        label = str(label.item())
        x = torch.squeeze(x).numpy()
        x = np.transpose(x, (1, 2, 0))
        ax = fig.add_subplot(9, 9, i+1, xticks=[], yticks=[])
        ax.imshow(x) #imshowの時の次元は(H, W, RGB)
        #ax.set_title(label, fontsize=8)
    
        i += 1
        if i >= 81:
            break
    plt.show(block=False)
    plt.pause(7.0)     #-->time.sleep()だと、plt.show()自体に結構な時間がかかるので、うまく表示されない。
    plt.close()

# Random Zoom Erasing
transform_zoom_erase = transforms.Compose([transforms.CenterCrop(size=100),transforms.Resize(120), transforms.RandomErasing(p = 0.8, scale = (0.1, 0.15), ratio = (0.3, 3.3))])

dataset_zoom_erase = torch.utils.data.TensorDataset(torch.Tensor(data['Type1']), transform_zoom_erase(image/255.0))

# 画像表示
fig = plt.figure(num='Zoom Erase', figsize=(12, 20))

dataloader_zoom_erase = torch.utils.data.DataLoader(dataset_zoom_erase, batch_size=1, shuffle=True)


def show_zoom_erase():
    i = 0
    for _, x in dataloader_zoom_erase:
        x = torch.squeeze(x).numpy()
        x = np.transpose(x, (1, 2, 0))
        ax = fig.add_subplot(9, 9, i+1, xticks=[], yticks=[])
        ax.imshow(x) #imshowの時の次元は(H, W, RGB)
    
        i += 1
        if i >= 81:
            break
    plt.show(block=False)
    plt.pause(7.0)
    plt.close()


#-----訓練検証用のデータセットの作成-----#

dataset_trainval = torch.utils.data.ConcatDataset([dataset_zoom_erase, dataset_zoom, dataset_rotate, dataset_rotate_shift, dataset_flip, dataset_shift_1, dataset_shift_2, dataset])
#train validationに分割
train_dataset, valid_dataset = torch.utils.data.random_split(dataset_trainval, [len(dataset_trainval)-140, 140])

if __name__ == '__main__':
    show_rotate_shift()
    show_zoom_erase()