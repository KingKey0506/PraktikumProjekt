import torch
import torch.nn as n
import numpy as np
from torch.nn import functional as f
from torchmetrics import Accuracy,MeanMetric
from tqdm import tqdm
import resnet.resres50 as res50
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision.transforms as trans
import torch.optim.lr_scheduler as lrsch

import test
''' resnet'''
b1 = n.Sequential(n.Conv2d(1,64, kernel_size=3, stride=1, padding=1),n.BatchNorm2d(64),n.ReLU(),#7×7 conv，stride 2 → BatchNorm → ReLU
                                                n.MaxPool2d(kernel_size=3, stride=2 ,padding=1))
b2 = n.Sequential(*res50.res_net(64, 64, 2, havefirst=True))
b3 = n.Sequential(*res50.res_net(64, 128, 2 ,havefirst=False))
b4 = n.Sequential(*res50.res_net(128, 256, 2, havefirst=False))
b5 = n.Sequential(*res50.res_net(256, 512, 2, havefirst=False))
net18 = n.Sequential(b1, b2, b3, b4, b5,n.AdaptiveAvgPool2d((1,1)), n.Flatten(),n.Linear(512, 7))#linnear 7 emotion type


'''resnet50'''
b1 = n.Sequential(n.Conv2d(1,64, kernel_size=7, stride=2, padding=1),n.BatchNorm2d(64),n.ReLU(),#7×7 conv，stride 2 → BatchNorm → ReLU
                                                n.MaxPool2d(kernel_size=3, stride=2 ,padding=1))
b2 = n.Sequential(*res50.resnet50(64, 64, 3, 1))
b3 = n.Sequential(*res50.resnet50(256, 128, 4,first_stride=2 ))
b4 = n.Sequential(*res50.resnet50(512, 256, 6,first_stride=2))
b5 = n.Sequential(*res50.resnet50(1024, 512, 3,first_stride=2))
net50 = n.Sequential(b1, b2, b3, b4, b5,n.AdaptiveAvgPool2d((1,1)), n.Flatten(),n.Dropout(0.5),n.Linear(2048, 7))# add
'''train'''
def train (net, train_it, test_it, num_epoche, lernrat, device ,writer, savePath):
    
    def init_weight (mo):
        if type(mo) == n.Linear or type(mo) == n.Conv2d:
            n.init.xavier_uniform_(mo.weight)
    
    # Handle device specification (cuda:0, cuda:1, etc.)
    if isinstance(device, str) and device.startswith('cuda'):
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device('cpu')
            print("CUDA not available, falling back to CPU")
    else:
        device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print('train on',device)
    net.to(device)
    #optimiz = torch.optim.SGD(net.parameters(), lr=lernrat,  momentum=0.9)
    optimiz = torch.optim.AdamW(net.parameters(), lr=lernrat, betas=(0.9, 0.95), weight_decay=1e-4)
    warmup = lrsch.LinearLR(optimiz, start_factor=0.1, total_iters=5)
    #cosine = lrsch.CosineAnnealingLR(optimiz, T_max=95, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiz, mode='min', factor=0.2, patience=5)
    loss = n.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean')# with sofmax
    train_avloss, train_avacu, test_accu, test_avloss =[], [], [], []
    for epoche in tqdm(range(num_epoche)):
        metric = res50.Accumulator(3)
        net.train()
        batch_losses = []
        batch_accs = []
        for i, (X,Y) in tqdm(enumerate(train_it)):
            X, Y = X.to(device),Y.to(device),
            optimiz.zero_grad()
            y_had = net(X)
            los = loss(y_had,Y)
            los.backward()
            optimiz.step()# change net.parameter
            preds = y_had.argmax(dim=1)
            correct = (preds == Y).sum().item()  
            metric.add(los*X.shape[0], correct, X.shape[0])
            batch_losses.append(los.item())
            batch_accs.append(correct / X.shape[0])
        
        avloss = metric[0] / metric[1]
        avaccu = metric[1] / metric[2]
        acc,acc_los = eaccuracy_gpu(net, test_it)
        
        train_avloss.append(avloss)
        train_avacu.append(avaccu)
        test_accu.append(acc)
        test_avloss.append(acc_los)
        print(acc)
        
        writer.add_scalar('Loss/train', avloss, epoche)
        writer.add_scalar('Accuracy/train', avaccu,epoche)
        writer.add_scalar('Accuracy/test', acc, epoche)
        writer.add_scalar('Loss/test', acc_los, epoche)
        writer.flush()
        if acc > bestAccuracy:
            bestAccuracy = acc
            torch.save(net.state_dict(), savePath)
            print(f"Model saved to {savePath}")
        if epoche < 5:
            warmup.step()
            print(f"Epoch {epoche}: lr = {scheduler.get_last_lr()[0]:.3e}")
        else:
            scheduler.step(acc_los)
            print(f"Epoch {epoche}: lr = {optimiz.param_groups[0]['lr']:.3e}")
    return train_avloss,train_avacu,test_accu,test_avloss
'''sets setting'''
class fersets(torch.utils.data.Dataset):
    def __init__(self,df,transform =None,randomcut = False,cut_padding = 12, reshape =None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.pixels = df["pixels"].values
        self.emotions = df["emotion"].values.astype(np.int64)
        self.transform = transform
        self.reshape = reshape
        self.randomcut = randomcut
        if randomcut:
            self.random_crop = trans.RandomCrop(
                size=(48, 48),
                padding=cut_padding,
                pad_if_needed=True
            )
        else:
            self.random_crop = None
        if reshape is not None:
            self.resize = trans.Resize((reshape, reshape))
        else:
            self.resize = None
    def __len__(self):
        return len(self.emotions)
    def __getitem__(self,idx):
        
        arry  = np.fromstring(self.pixels[idx], sep = " ",dtype=np.uint8)
        img = arry.reshape(48,48)
        img = torch.from_numpy(img).unsqueeze(0).float()/255#bchw
        label = int(self.emotions[idx])
        if self.resize :
            img = self.resize(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.randomcut is not False:
            img = self.random_crop(img)
        if self.resize :
            img = self.resize(img)
        return img, label

'''use gpu accu'''
def eaccuracy_gpu(net, data_it, device = None, num_classes=7):
    if device is None:
        device = next(net.parameters()).device
    else:
        net = net.to(device)
    if isinstance(net, torch.nn.Module):
        net.eval()#aviod dropout
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    metric_los = MeanMetric().to(device)
    loss = n.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean')
    with torch.no_grad():# no grandiant 
        for X,Y in data_it:#y label
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            Y = Y.to(device)
            log = net(X)
            los = loss(log, Y)
            metric_los.update(los)
            preds = log.argmax(dim=1)#Prediction Category
            metric.update(preds, Y)
        return metric.compute().item(),metric_los.compute().item()

