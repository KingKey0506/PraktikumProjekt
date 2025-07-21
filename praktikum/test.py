import train_cv 
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as n
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as trans
import random
import torchvision.transforms.functional as tf
import datetime
import os

from torch.utils.tensorboard import SummaryWriter
train_transform = trans.Compose([
    trans.RandomResizedCrop(48, scale=(0.8, 1.0)),  
    trans.RandomHorizontalFlip(),                  
    trans.RandomRotation(15),                       
])


def augment_trans(tensor_img, turn_angel):
    if random.random()<0.5:
        tensor_img = tf.hflip(tensor_img)
        if random.random()<0.6:
            tensor_img = tf.rotate(tensor_img, angle= random.uniform(-turn_angel, turn_angel))
    if random.random()<0.7:
        tensor_img = tf.adjust_contrast(tensor_img, contrast_factor=0.8)
    return tensor_img


        
def to_loader(df, batch_size, reshape,shuffle):
    ds = train_cv.fersets(df.reset_index(drop=True),transform=lambda img: augment_trans(img,15) ,randomcut=False,reshape=reshape)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def main(epoche=10, lr=0.05, device="cuda", logroot="runs", net="resnet50"):
    
    '''parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.05)
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--logroot",    type=str,   default="runs")
    args = parser.parse_args()'''

    fer2013 = pd.read_csv("resnet/fer2013.csv")

    print(fer2013.shape)
    #print(fer2013.head())

    CLASS_LABELS  = ['Anger','Disgust','Fear','Happy','Neutral','Sadness','Surprise']
    train_data = fer2013.sample(frac=1)#shuffle keep index
    train_data = train_data[train_data["Usage"] == "Training"]

    #print (train_data.sample(1).iloc[0])#check
    train_df, val_f = train_test_split(train_data,test_size=0.2,stratify=train_data["emotion"],random_state=42)
    train_it = to_loader(train_df, batch_size=64, reshape= 224,shuffle=True)

    test_it = to_loader(val_f, batch_size=64, reshape= 224 ,shuffle=False)

    now     = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_na  = f"resnet18-lr{lr}-{now}"
    log_dir = os.path.join(logroot, run_na)
    os.makedirs(log_dir, exist_ok=True)
    writer  = SummaryWriter(log_dir=log_dir)

    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = train_cv.net50.to(device)
    X_batch, Y_batch = next(iter(train_it))
    X = X_batch.to(device)
    Y = Y_batch.to(device)

    
    with torch.no_grad():
        for name, module in net.named_children():
            X = module(X)
            print(f"{name:<5} {tuple(X.shape)}")

    
    #train_avloss,train_avacu,test_accu = train_cv.train(train_cv.net, train_it, test_it, 10, 0.035, 'cuda')
    train_avloss, train_avaccu, test_accu, test_avloss= train_cv.train(
        net       = train_cv.net,
        train_it  = train_it,
        test_it   = test_it,
        num_epoche= epoche,    # 
        lernrat   = lr,        # 
        device    = device,    # 
        writer    = writer          # 
    )
    writer.close()

    print("train_avloss:",train_avloss)
    print("train_avaccu:",train_avaccu)
    print("test_accu:",test_accu)
    print("test_accu:",test_avloss)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exam , label = next(iter(test_it))# Visualize a batch and its predictions
    exam = exam.to(device)
    label = label.to(device)
    with torch.no_grad():
        pre_after = train_cv.net50(exam)# TODO
    label_pre = torch.argmax(pre_after,dim=1)
    exam_cpu     = exam.cpu()
    label_cpu    = label.cpu()
    label_pre_cpu= label_pre.cpu()
    plt.figure(figsize=(10,10))
    for pic in range(16):
        plt.subplot(4,4,pic+1)
        plt.imshow(exam[pic].cpu().numpy().reshape(224,224), cmap='gray')
        plt.title(f'pred:{label_pre[pic]}, label:{label[pic]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig( os.path.join(log_dir, f"predictions.png {now}"))
if __name__ == "__main__":
    main()

'''
def print_module_shapes(x, module, prefix=""):
    for name, child in module.named_children():
        x = child(x)
        print(f"{prefix}{name:<25} {child.__class__.__name__:<20} {tuple(x.shape)}")
        # 如果这个 child 又是个容器，就递归进去
        if isinstance(child, n.Sequential):
            print_module_shapes(x, child, prefix + "  ")

# 取一个 batch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = train_cv.net.to(device)
X_batch, Y_batch = next(iter(train_it))
X = X_batch.to(device)
Y = Y_batch.to(device)
print("X_batch.shape:", X_batch.shape)
print(Y_batch.shape)
# 调用
print_module_shapes(X, net)
'''