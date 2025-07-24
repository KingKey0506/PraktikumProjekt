import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm #progress/loading bars
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from collections import deque, Counter
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test
#from skimage.transform import resize
#import dlib
from collections import deque, Counter

# Emotion labels -- emotionto index and vice versa
emotions = ['happy', 'surprise', 'sad', 'angry', 'disgust', 'fear']
EmotionToIndex = {emotion: idx for idx, emotion in enumerate(emotions)}
IndexToEmotion = {idx: emotion for emotion, idx in EmotionToIndex.items()}

def ResolvePath(path):
    #Resolve relative paths to absolute paths
    if os.path.isabs(path):
        return path
    else:
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_path = os.path.join(script_dir, path)
        return os.path.normpath(resolved_path)


# Custom CNN

class EmotionDetectionCNN(nn.Module):
    def __init__(self, EmotionAnzahl=6):
        super(EmotionDetectionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, EmotionAnzahl)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x # end of first prediction

class FaceEmotionDataset(Dataset):
    def __init__(self, DataFolder, transform=None, cache=True):
        self.DataFolder = DataFolder
        self.transform = transform
        self.cache = cache
  # set where to save the processed images and their emotions
        self.CacheDataPath = os.path.join(DataFolder, 'cached_data.npy')
        self.CacheEmotionsPath = os.path.join(DataFolder, 'cached_labels.npy')
        
        if self.cache and os.path.exists(self.CacheDataPath) and os.path.exists(self.CacheEmotionsPath):
            # Load cached data
            print(f"Loading cached data from {DataFolder}...")
            self.data = np.load(self.CacheDataPath)
            self.labels = np.load(self.CacheEmotionsPath)
            print(f"Loaded {len(self.data)} cached samples")
        else:  #manual caching
            
            print(f"Processing and caching data from {DataFolder}...")
            self.data = []
            self.labels = []
            
            for emotion in emotions:
                EmotionDirectory = os.path.join(DataFolder, emotion)
                if not os.path.exists(EmotionDirectory): # Could be a spelling mistake or wrong data set location etc.
                    print(f"Warning: {EmotionDirectory} not found, skipping")
                    continue
                
                print(f"Processing {emotion} images...")
                ImageFiles = [f for f in os.listdir(EmotionDirectory) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for ImageFile in tqdm(ImageFiles, desc=f"Loading {emotion}"):
                    ImagePath = os.path.join(EmotionDirectory, ImageFile)
                    try:
                        Image= cv2.imread(ImagePath) #if image is unreadable
                        if Image is None:
                            continue
                        Image= cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
                        Image= cv2.resize(Image, (64, 64)) # gegeben
                        Image= Image.astype(np.float32) / 255.0 # 0.0-1.0
                        Image= np.transpose(Image, (2, 0, 1))  # rearranges the shape of the image HWC -> CHW (PyTorch format)
                        
                        self.data.append(Image)
                        self.labels.append(EmotionToIndex[emotion])
                    except Exception as e:
                        print(f"Error processing {ImagePath}: {e}")
                        continue
            
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            
        # no need to reload next time   
            if self.cache:
                np.save(self.CacheDataPath, self.data)
                np.save(self.CacheEmotionsPath, self.labels)
                print(f"Cached {len(self.data)} samples to {self.CacheDataPath}")
    
    def __len__(self): #how many items
        return len(self.data)
    
    def __getitem__(self, idx): #how to get one specific image and its label
        Image= self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            
            Image= np.transpose(Image, (1, 2, 0)) # Convert back to HWC for transforms
            Image= (Image* 255).astype(np.uint8) # original scale
            Image= self.transform(Image)
        else:
            # Convert to tensor
            Image= torch.FloatTensor(Image)
        
        return Image, label

def TrainFromScratch(TrainingDirectory, ValidationDirectory=None, epochs=30, batch_size=64, lr=0.001, savePath='emotion_cnn_scratch.pth'):
    now     = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_na  = f"CNN-lr{lr}-{now}"
    log_dir = os.path.join('runlog', run_na)
    os.makedirs(log_dir, exist_ok=True)
    writer  = SummaryWriter(log_dir=log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #gpu else CPU 
    print(f"Using device: {device}")
    #import os
    print(f"Current working directory: {os.getcwd()}")
    
    TrainingDirectory = ResolvePath(TrainingDirectory) # absolute paths so wie oben
    if ValidationDirectory:
        ValidationDirectory = ResolvePath(ValidationDirectory)
    
    print(f"Training directory: {TrainingDirectory}")
    print(f"Training directory exists: {os.path.exists(TrainingDirectory)}")
    if ValidationDirectory:
        print(f"Validation directory: {ValidationDirectory}")
        print(f"Validation directory exists: {os.path.exists(ValidationDirectory)}")
    
    transform = transforms.Compose([
        transforms.ToTensor(), #pytorch tensors ahnlich zu Matrizen 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    TrainDataset = FaceEmotionDataset(TrainingDirectory, transform=transform) # the part where the learning happens
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True) # shuffle helps improvment 
    if ValidationDirectory:
        ValDataset = FaceEmotionDataset(ValidationDirectory, transform=transform)
        ValDataloader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False) # hier kein Shuffle 
    else:
        ValDataloader = None
    model = EmotionDetectionCNN(EmotionAnzahl=6).to(device) # my model 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    bestAccuracy = 0
    for epoch in range(epochs):
        model.train() #training mode 
        totalLoss, correct, total = 0, 0, 0
        for Images, labels in tqdm(TrainDataLoader, desc=f"Epoch {epoch+1}/{epochs}"): # loops in batches and gives tqdm
            Images, labels = Images.to(device), labels.to(device)
            optimizer.zero_grad() # gradiant else will build up 
            outputs = model(Images) # forward pass to send through the cnn Layers 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() #update weights 
            totalLoss += loss.item() # gets it from the pytorch tensor
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        trainingAcc = correct / total
        writer.add_scalar("Loss/train", totalLoss / len(TrainDataLoader), epoch)
        writer.add_scalar("Accuracy/train", trainingAcc, epoch)
        print(f"Train Loss: {totalLoss/len(TrainDataLoader):.4f}, Train Acc: {trainingAcc:.4f}")
        if ValDataloader:
            model.eval() #evaluation mode 
            valCorrect, valTotal = 0, 0
            val_batch = 0
            valTotalLoss = 0
            with torch.no_grad(): # dont track because my goal here is not to train 
                for Images, labels in ValDataloader: # work in batches 
                    Images, labels = Images.to(device), labels.to(device)
                    outputs = model(Images)
                    _, preds = torch.max(outputs, 1)
                    valCorrect += (preds == labels).sum().item()
                    valloss = criterion(outputs, labels)
                    valTotalLoss += valloss.item()
                    valTotal += labels.size(0)
                    val_batch += 1
            valAcc = valCorrect / valTotal
            val_loss = valTotalLoss / val_batch
            writer.add_scalar("Accuracy/Test", valAcc, epoch)
            writer.add_scalar("Loss/Test", val_loss , epoch)
            print(f"Val Acc: {valAcc:.4f}")
            if valAcc > bestAccuracy:
                bestAccuracy = valAcc
                torch.save(model.state_dict(), savePath)
                print(f"Model saved to {savePath}")
        else:
            torch.save(model.state_dict(), savePath)
    print("Training complete.")
    return model


def predictToCSV(ModelPath, FolderPath, output_CSV='results.csv'):
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelPath = ResolvePath(ModelPath)
    FolderPath = ResolvePath(FolderPath)
    print(f"Loading model from: {ModelPath}")
    print(f"Classifying images in: {FolderPath}")
    model = EmotionDetectionCNN(EmotionAnzahl=6).to(device)
    model.load_state_dict(torch.load(ModelPath, map_location=device))
    model.eval() #to make sure it's as consestiant as possible 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    results = []
    imageFiles = []
    for root, _, files in os.walk(FolderPath):
        for ImageFile in files:
            if ImageFile.lower().endswith(('.png', '.jpg', '.jpeg')):
                imageFiles.append(os.path.join(root, ImageFile))
    for ImagePath in tqdm(sorted(imageFiles), desc='Classifying images'):
        Image= cv2.imread(ImagePath)
        Image= cv2.cvtColor(Image, cv2.COLOR_BGR2RGB) 
        Image= cv2.resize(Image, (64, 64)) # das sind ein paar Formalitäten 
        ImageTensor  = transform(Image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(ImageTensor )
            probs = F.softmax(output, dim=1).cpu().numpy()[0] #softmax for probablities
    
        relativePath = os.path.relpath(ImagePath, start=os.getcwd()) # path to current directory
        relativePath = '/' + relativePath.replace('\\', '/').replace('\\', '/')
        row = {'filepath': relativePath}
        for i, emotion in enumerate(emotions):
            row[emotion] = f"{probs[i]:.2f}"
        results.append(row)
    columns = ['filepath'] + emotions
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_CSV, index=False)
    print(f"Results saved to {output_CSV}")

# ----------------------
# 3. Video Demo (classify each frame, overlay saliency, save video)
# ----------------------
def overlay_saliency_on_frame(model, frame, device):
    # Use the full frame for prediction (no mouth crop)
    Image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image_resized = cv2.resize(Image, (64, 64))
    overlay_base = frame
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(Image_resized).unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    # Forward
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    max_prob = float(np.max(probs))
    # Saliency
    model.zero_grad()
    output[0, pred_idx].backward()
    saliency = input_tensor.grad.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    # Apply Gaussian blur for smoothness
    saliency = cv2.GaussianBlur(saliency, (11, 11), 0)
    # Normalize to 0-255
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = np.uint8(255 * saliency)
    saliency = cv2.resize(saliency, (overlay_base.shape[1], overlay_base.shape[0]))
    # Apply colormap
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    # Overlay: more weight to original image, less to heatmap
    overlay = cv2.addWeighted(overlay_base, 0.7, heatmap, 0.3, 0)
    return overlay, pred_idx, max_prob

def ProcessVideo(ModelPath, VideoPath, OutputPath='output_video.avi'):
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionDetectionCNN(EmotionAnzahl=6).to(device)
    model.load_state_dict(torch.load(ModelPath, map_location=device))
    model.eval()
    cap = cv2.VideoCapture(VideoPath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    frame_count = 0
    pred_buffer = deque(maxlen=10)  # Buffer for smoothing predictions
    conf_buffer = deque(maxlen=10)  # Buffer for confidences
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc='Processing video frames')
    neutral_threshold = 0.5
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay, pred_idx, max_prob = overlay_saliency_on_frame(model, frame, device)
        pred_buffer.append(pred_idx)
        conf_buffer.append(max_prob)
        # Use the most common prediction in the buffer
        if len(pred_buffer) > 0:
            most_common_pred = Counter(pred_buffer).most_common(1)[0][0]
            # Calculate average confidence for the most common prediction
            relevant_confs = [conf for pred, conf in zip(pred_buffer, conf_buffer) if pred == most_common_pred]
            avg_conf = np.mean(relevant_confs) if relevant_confs else 0.0
            if avg_conf < neutral_threshold:
                label = 'neutral'
            else:
                label = IndexToEmotion[most_common_pred]
        else:
            label = 'neutral'
        cv2.putText(overlay, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if out is None:
            out = cv2.VideoWriter(OutputPath, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
        out.write(overlay)
        frame_count += 1
        pbar.update(1)
    cap.release()
    if out:
        out.release()
    pbar.close()
    print(f"Processed {frame_count} frames. Output saved to {OutputPath}")

# ----------------------
# 4. Webcam Demo (real-time)
# ----------------------
def overlay_saliency_on_frame(model, frame, device, face_roi=None):
    # 1. cut ROI
    if face_roi is not None:
        x1, y1, x2, y2 = face_roi
        roi = frame[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
        roi = frame
    #print("roi.shape:", roi.shape)  # <-- ①
    # 2.  RGB + resize
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(rgb, (64, 64))
    #print("img_resized.shape:", img_resized.shape)  # <-- ②

    # 3. ToTensor + Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    input_tensor = transform(img_resized).unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    #print("input_tensor.shape:", input_tensor.shape)  # <-- ③

    # 4. Forward + backward
    output = model(input_tensor)
    pred_idx = int(output.argmax(dim=1))
    model.zero_grad()
    output[0, pred_idx].backward()

    grad = input_tensor.grad.abs().squeeze().cpu().numpy()
    #print("grad.shape:", grad.shape)  # <-- ④

    saliency = np.max(grad, axis=0)
    #print("saliency.shape:", saliency.shape)  # <-- ⑤

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = np.uint8(255 * saliency)
    saliency = cv2.resize(saliency, (x2-x1, y2-y1))

    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    #print("heatmap.shape:", heatmap.shape)  # <-- ⑥

    overlay = frame.copy()
    #print("frame[y1:y2, x1:x2].shape:", frame[y1:y2, x1:x2].shape)  # <-- ⑦
    overlay[y1:y2, x1:x2] = cv2.addWeighted(
        frame[y1:y2, x1:x2], 0.7,
        heatmap,            0.3,
        0
    )
    # 7.retern overlay、 idx、max_prob
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    max_prob = float(probs[pred_idx])
    
    return overlay, pred_idx, max_prob
def WebcamDemo(ModelPath):
    #prototxt = ResolvePath('deploy.prototxt')
    #caffemodel = ResolvePath('res10_300x300_ssd_iter_140000.caffemodel') open in same dir
    caffemodel = r"E:\githb\PraktikumProjekt\praktikum\res10_300x300_ssd_iter_140000.caffemodel"
    prototxt = r"E:\githb\PraktikumProjekt\praktikum\deploy.prototxt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionDetectionCNN(EmotionAnzahl=6).to(device)
    model.load_state_dict(torch.load(ModelPath, map_location=device, weights_only=True))
    model.eval()
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
    print("Press 'q' to quit.")
    print("Proto exists:", prototxt, os.path.exists(prototxt))
    print("Model exists:", caffemodel, os.path.exists(caffemodel))
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("RAW", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                     (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        best_conf = 0
        best_box = None
        for i in range(detections.shape[2]):
            conf = float(detections[0,0,i,2])
            if conf > 0.6 and conf > best_conf:
                best_conf = conf
                box = detections[0,0,i,3:7] * np.array([w, h, w, h])
                best_box = box.astype(int)

        
        if best_box is not None:
            x1,y1,x2,y2 = best_box
          
            overlay, pred_idx, max_prob = overlay_saliency_on_frame(
                model, frame, device,
                face_roi=(x1,y1,x2,y2)  
            )
        else:
            overlay, pred_idx ,max_prob= overlay_saliency_on_frame(model, frame, device)
        label = IndexToEmotion[pred_idx]
        cv2.putText(overlay, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('Webcam Emotion Recognition', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
#----------------------------------------------------------------------
# start of implementing interactive mode

if __name__ == '__main__': # only when executed directly not through import
    import argparse
    import sys
   # default paths (examples for user)
    defaultModelSavePath = 'E:/githb/PraktikumProjekt/emotion_model.pth'
    defaultTrainingImagesDir = r'E:\githb\PraktikumProjekt\archive\train'
    defaultTestImagesDirectory = r'E:\githb\PraktikumProjekt\archive\test'
 
    
    if len(sys.argv) == 1: # check if running on an IDE

        # if yes....
        print("=" * 60)
        print("EMOTION CNN - INTERACTIVE MENU")
        print("=" * 60)
        print("1. Train model from scratch")
        print("2. Batch classify images in folder to CSV")
        print("3. Process video with emotion classification")
        print("4. Webcam real-time demo")
        print("5. Exit")
        print("=" * 60)
        
        while True:
            try:
                UserInput = input("Enter your User Input (1-5): ").strip()
                
                if UserInput == '1':
                    print("\n=== TRAINING MODEL ===")
                    # input your own training directory like in the example given
                    TrainingDirectory = input(f"Training directory (default: {defaultTrainingImagesDir}): ").strip() 
                    if not TrainingDirectory:
                        TrainingDirectory = defaultTrainingImagesDir
                    
                    ValidationDirectory = input(f"Validation directory (default: {defaultTestImagesDirectory}): ").strip()
                    if not ValidationDirectory:
                        ValidationDirectory = defaultTestImagesDirectory
                    
                    # Je größer die Anzahl an Epochen, desto höher die "accuracy"
                    epochs = input("Number of epochs (default: 30): ").strip()
                    epochs = int(epochs) if epochs else 30
                    
                    batch_size = input("Batch size (default: 64): ").strip() # process images in groups of 64
                   
                    batch_size = int(batch_size) if batch_size else 64
                    
                    lr = input("Learning rate (default: 0.001): ").strip() # smaller lr for smaller adjustment of weights in response to error
                    lr = float(lr) if lr else 0.001

                    # where should the model be saved? Create new file before executing 
                    savePath = input(f"Model save path (default: {defaultModelSavePath}): ").strip() 
                    if not savePath:
                        savePath = defaultModelSavePath

                    # summarize the data the user passed
                    print(f"\nStarting training with:")
                    print(f"Train dir: {TrainingDirectory}")
                    print(f"Val dir: {ValidationDirectory}")
                    print(f"Epochs: {epochs}")
                    print(f"Batch size: {batch_size}")
                    print(f"Learning rate: {lr}")
                    print(f"Save path: {savePath}")

                    #call function from line "142"
                    TrainFromScratch(TrainingDirectory, ValidationDirectory, epochs, batch_size, lr, savePath) 
                    
                elif UserInput == '2':
                    print("\n=== BATCH CLASSIFICATION ===")
                    ModelPath = input(f"Model path (default: {defaultModelSavePath}): ").strip()
                    if not ModelPath:
                        ModelPath = defaultModelSavePath
                    FolderPath = input(f"Folder path to classify (default: {defaultTestImagesDirectory}): ").strip()
                    if not FolderPath:
                        FolderPath = defaultTestImagesDirectory
                    output_CSV = input("Output CSV filename (default: results.csv): ").strip()
                    if not output_CSV:
                        output_CSV = 'results.csv'
                    
                    predictToCSV(ModelPath, FolderPath, output_CSV) # label all images and write the results to a CSV

                elif UserInput == '1.5':
                    print(f"set model parameters in shape --epoche  --lr --net('resnet50') --device 'cuda'--logroot 'runs'")
                    
                    epochs     = input("Number of epochs (default 10): ").strip()
                    epochs     = int(epochs) if epochs else 10
                    lr         = input("Learning rate (default 0.05): ").strip()
                    lr         = float(lr) if lr else 0.05
                    net        = input("Net (resnet50/resnet18, default resnet50): ").strip()
                    net        = net if net else "resnet50"
                    device     = input("Device (cuda/cpu, default cuda): ").strip()
                    device     = device if device else "cuda"
                    logroot    = input("Logroot (default runs): ").strip()
                    logroot    = logroot if logroot else "runs"
                    #  test.py  main
                    test.main(
                        epoche=epochs,
                        #batch_size=64,  # default batch size
                        lr=lr,
                        device=device,
                        logroot=logroot,
                        net=net
                    )    
                elif UserInput == '3':
                    print("\n=== VIDEO PROCESSING ===")
                    ModelPath = input(f"Model path (default: {defaultModelSavePath}): ").strip()
                    if not ModelPath:
                        ModelPath = defaultModelSavePath
                    VideoPath = input("Input video path: ").strip()
                    # Where should the processed video with predictions be saved
                    OutputPath = input("Output video path (default: output_video.avi): ").strip()
                    if not OutputPath:
                        OutputPath = 'output_video.avi'
                    
                    ProcessVideo(ModelPath, VideoPath, OutputPath)
                    
                elif UserInput == '4':
                    print("\n=== WEBCAM DEMO ===")
                    ModelPath = input(f"Model path (default: {defaultModelSavePath}): ").strip()
                    if not ModelPath:
                        ModelPath = defaultModelSavePath
                    print("Starting webcam demo... Press 'q' to quit.")
                    WebcamDemo(ModelPath)
                    
                elif UserInput == '5':
                    print("Exiting...")
                    break
                    
                else:
                    print("Invalid UserInput. Please enter 1-5.") # start from the beginning because of mistake in user input
                    
                print("\n" + "=" * 60)
                print("1. Train model from scratch")
                print("2. Batch classify images in folder to CSV")
                print("3. Process video with emotion classification")
                print("4. Webcam real-time demo")
                print("5. Exit")
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                # Catch and display any unexpected error
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")
    
    else:
        
        parser = argparse.ArgumentParser(description='Emotion CNN - Train, Batch Classify, Video Demo, Webcam Demo')
        subparsers = parser.add_subparsers(dest='command')

        # Train
        train_parser = subparsers.add_parser('train', help='Train model from scratch')
        train_parser.add_argument('--TrainingDirectory', default=defaultTrainingImagesDir)
        train_parser.add_argument('--ValidationDirectory', default=defaultTestImagesDirectory)
        train_parser.add_argument('--epochs', type=int, default=30)
        train_parser.add_argument('--batch_size', type=int, default=64)
        train_parser.add_argument('--lr', type=float, default=0.001)
        train_parser.add_argument('--savePath', default=defaultModelSavePath)

        # for Batch classification
        classify_parser = subparsers.add_parser('classify', help='Batch classify images in a folder and output CSV')
        classify_parser.add_argument('--ModelPath', default=defaultModelSavePath)
        classify_parser.add_argument('--FolderPath', default=defaultTestImagesDirectory)
        classify_parser.add_argument('--output_CSV', default='results.csv')

        # for Video demo
        video_parser = subparsers.add_parser('video', help='Process a video, classify, overlay saliency, save video')
        video_parser.add_argument('--ModelPath', default=defaultModelSavePath, required=False)
        video_parser.add_argument('--VideoPath', required=True)
        video_parser.add_argument('--OutputPath', default='output_video.avi')

        # for Webcam demo
        webcam_parser = subparsers.add_parser('webcam', help='Webcam real-time demo')
        webcam_parser.add_argument('--ModelPath', default=defaultModelSavePath, required=False)

        args = parser.parse_args()
        if args.command == 'train':
            TrainFromScratch(args.TrainingDirectory, args.ValidationDirectory, args.epochs, args.batch_size, args.lr, args.savePath)
        elif args.command == 'classify':
            predictToCSV(args.ModelPath, args.FolderPath, args.output_CSV)
        elif args.command == 'video':
            ProcessVideo(args.ModelPath, args.VideoPath, args.OutputPath)
        elif args.command == 'webcam':
            WebcamDemo(args.ModelPath)
        else:
            parser.print_help() 