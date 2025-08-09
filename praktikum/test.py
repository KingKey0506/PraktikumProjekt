import train_cv 
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as n
import numpy as np
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

def main(epoche=None, batch_size=None, lr=None, device=None, logroot=None, net=None, data_path=None, reshape=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.05)
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--logroot",    type=str,   default="runs")
    parser.add_argument("--net",        type=str,   default="net50")
    parser.add_argument('--data-path',  type=str,   default="resnet/fer2013.csv", help='Path to FER2013 dataset')
    parser.add_argument('--reshape',    type=int,   default=224)
    args = parser.parse_args()
    
    # Override with direct parameters if provided (for interactive use)
    if epoche is not None:
        args.epochs = epoche
    if batch_size is not None:
        args.batch_size = batch_size  
    if lr is not None:
        args.lr = lr
    if device is not None:
        args.device = device
    if logroot is not None:
        args.logroot = logroot
    if net is not None:
        args.net = net
    if data_path is not None:
        args.data_path = data_path
    if reshape is not None:
        args.reshape = reshape
    
    # Resolve relative path to absolute path
    if not os.path.isabs(args.data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_path = os.path.join(script_dir, args.data_path)
        args.data_path = os.path.normpath(args.data_path)
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found!")
        print("Please make sure dataset is in the current directory or specify --data-path")
        print(f"Looking for file at: {args.data_path}")
        return
    
    print(f"Starting training with parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Network: {args.net}")
    print(f"  Device: {args.device}")
    print(f"  Data Path: {args.data_path}")
    fer2013 = pd.read_csv(args.data_path)

    print(fer2013.shape)
    #print(fer2013.head())
    picshape =args.reshape
    CLASS_LABELS  = ['Anger','Disgust','Fear','Happy','Neutral','Sadness','Surprise']
    train_data = fer2013.sample(frac=1)#shuffle keep index
    train_data = train_data[train_data["Usage"] == "Training"]

    #print (train_data.sample(1).iloc[0])#check
    train_df, val_f = train_test_split(train_data,test_size=0.2,stratify=train_data["emotion"],random_state=42)
    train_it = to_loader(train_df, batch_size=args.batch_size, reshape= picshape,shuffle=True)

    test_it = to_loader(val_f, batch_size=args.batch_size, reshape= picshape ,shuffle=False)

    now     = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_na  = f"-net{args.net}-lr{args.lr}-{now}"
    log_dir = os.path.join(args.logroot, run_na)
    os.makedirs(log_dir, exist_ok=True)
    writer  = SummaryWriter(log_dir=log_dir)
    model_dir = os.path.join("weights", run_na)
    os.makedirs(model_dir, exist_ok=True)
    savePath = f"model_{args.net}_lr{args.lr}.pth"
    SAVE_dir = os.path.join(model_dir, savePath)
    total = 0
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')  # Specify GPU 1
            print(f"Using GPU 1: {torch.cuda.get_device_name(1)}")
        else:
            device = torch.device('cuda:0')  # Fallback to GPU 0 if only one GPU
            print(f"Only one GPU available, using GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    if args.net == "net50":
        net = train_cv.net50
    elif args.net == "net18":
        net = train_cv.net18
    else:
        raise ValueError(f"Unknown network type: {args.net}")
    net = net.to(device)
    X_batch, Y_batch = next(iter(train_it))
    X = X_batch.to(device)
    Y = Y_batch.to(device)

    
    with torch.no_grad():
        for name, module in net.named_children():
            X = module(X)
            print(f"{name:<5} {tuple(X.shape)}")

    
    #train_avloss,train_avacu,test_accu = train_cv.train(train_cv.net, train_it, test_it, 10, 0.035, 'cuda')
    device_str = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
    train_avloss, train_avaccu, test_accu, test_avloss= train_cv.train(
        net       = net,
        train_it  = train_it,
        test_it   = test_it,
        num_epoche= args.epochs,    # 
        lernrat   = args.lr,        # 
        device    = device_str,     # Use GPU 1 if available, otherwise GPU 0
        writer    = writer ,         # 
        savePath  = SAVE_dir  # Save model path
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
        pre_after = net(exam)
    label_pre = torch.argmax(pre_after,dim=1)
    exam_cpu     = exam.cpu()
    label_cpu    = label.cpu()
    label_pre_cpu= label_pre.cpu()
    plt.figure(figsize=(10,10))
    for pic in range(16):
        plt.subplot(4,4,pic+1)
        plt.imshow(exam[pic].cpu().numpy().reshape(picshape,picshape), cmap='gray')
        plt.title(f'pred:{label_pre[pic]}, label:{label[pic]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig( os.path.join(log_dir, f"predictions.png {now}.png"))
if __name__ == "__main__":
    main()

