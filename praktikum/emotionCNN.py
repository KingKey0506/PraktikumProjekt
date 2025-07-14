import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
import dlib
from collections import deque, Counter

# Emotion labels
EMOTIONS = ['happy', 'surprise', 'sad', 'angry', 'disgust', 'fear']
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

def resolve_path(path):
    """Resolve relative paths to absolute paths"""
    import os
    if os.path.isabs(path):
        return path
    else:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Resolve relative to script directory
        resolved_path = os.path.join(script_dir, path)
        return os.path.normpath(resolved_path)

# ----------------------
# 1. Custom CNN (from scratch)
# ----------------------
class SimpleEmotionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleEmotionCNN, self).__init__()
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
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------
# Dataset (reuse from main.py, simplified)
# ----------------------
class FacialImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, cache=True):
        self.data_dir = data_dir
        self.transform = transform
        self.cache = cache
        
        # Cache file paths
        self.cache_data_path = os.path.join(data_dir, 'cached_data.npy')
        self.cache_labels_path = os.path.join(data_dir, 'cached_labels.npy')
        
        if self.cache and os.path.exists(self.cache_data_path) and os.path.exists(self.cache_labels_path):
            # Load cached data
            print(f"Loading cached data from {data_dir}...")
            self.data = np.load(self.cache_data_path)
            self.labels = np.load(self.cache_labels_path)
            print(f"Loaded {len(self.data)} cached samples")
        else:
            # Process and cache data
            print(f"Processing and caching data from {data_dir}...")
            self.data = []
            self.labels = []
            
            for emotion in EMOTIONS:
                emotion_dir = os.path.join(data_dir, emotion)
                if not os.path.exists(emotion_dir):
                    print(f"Warning: {emotion_dir} not found, skipping...")
                    continue
                
                print(f"Processing {emotion} images...")
                img_files = [f for f in os.listdir(emotion_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_file in tqdm(img_files, desc=f"Loading {emotion}"):
                    img_path = os.path.join(emotion_dir, img_file)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (64, 64))
                        img = img.astype(np.float32) / 255.0
                        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                        
                        self.data.append(img)
                        self.labels.append(EMOTION_TO_IDX[emotion])
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
            
            # Convert to numpy arrays
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            
            # Save cache
            if self.cache:
                np.save(self.cache_data_path, self.data)
                np.save(self.cache_labels_path, self.labels)
                print(f"Cached {len(self.data)} samples to {self.cache_data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Convert back to (H, W, C) for transforms
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)
            img = self.transform(img)
        else:
            # Convert to tensor
            img = torch.FloatTensor(img)
        
        return img, label

# ----------------------
# 1. Training from scratch
# ----------------------
def train_from_scratch(train_dir, val_dir=None, epochs=30, batch_size=64, lr=0.001, save_path='emotion_cnn_scratch.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Debug: Print current working directory and check if paths exist
    import os
    print(f"Current working directory: {os.getcwd()}")
    
    # Resolve paths
    train_dir = resolve_path(train_dir)
    if val_dir:
        val_dir = resolve_path(val_dir)
    
    print(f"Training directory: {train_dir}")
    print(f"Training directory exists: {os.path.exists(train_dir)}")
    if val_dir:
        print(f"Validation directory: {val_dir}")
        print(f"Validation directory exists: {os.path.exists(val_dir)}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = FacialImageDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dir:
        val_dataset = FacialImageDataset(val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    model = SimpleEmotionCNN(num_classes=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        print(f"Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        if val_loader:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total
            print(f"Val Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        else:
            torch.save(model.state_dict(), save_path)
    print("Training complete.")
    return model

# ----------------------
# 2. Batch Folder Classification to CSV
# ----------------------
def classify_folder_to_csv(model_path, folder_path, output_csv='results.csv'):
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Resolve paths
    model_path = resolve_path(model_path)
    folder_path = resolve_path(folder_path)
    print(f"Loading model from: {model_path}")
    print(f"Classifying images in: {folder_path}")
    model = SimpleEmotionCNN(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    results = []
    # Recursively find all image files
    image_files = []
    for root, _, files in os.walk(folder_path):
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, img_file))
    for img_path in tqdm(sorted(image_files), desc='Classifying images'):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
        # Build row: filepath (with leading /), then probabilities as strings with two decimals
        rel_path = os.path.relpath(img_path, start=os.getcwd())
        rel_path = '/' + rel_path.replace('\\', '/').replace('\\', '/')
        row = {'filepath': rel_path}
        for i, emotion in enumerate(EMOTIONS):
            row[emotion] = f"{probs[i]:.2f}"
        results.append(row)
    # Ensure columns order
    columns = ['filepath'] + EMOTIONS
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# ----------------------
# 3. Video Demo (classify each frame, overlay saliency, save video)
# ----------------------
def overlay_saliency_on_frame(model, frame, device):
    # Use the full frame for prediction (no mouth crop)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (64, 64))
    overlay_base = frame
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_resized).unsqueeze(0).to(device)
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
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = cv2.resize(saliency, (overlay_base.shape[1], overlay_base.shape[0]))
    # Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(overlay_base, 0.6, heatmap, 0.4, 0)
    return overlay, pred_idx, max_prob

def process_video(model_path, video_path, output_path='output_video.avi'):
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleEmotionCNN(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    cap = cv2.VideoCapture(video_path)
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
                label = IDX_TO_EMOTION[most_common_pred]
        else:
            label = 'neutral'
        cv2.putText(overlay, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
        out.write(overlay)
        frame_count += 1
        pbar.update(1)
    cap.release()
    if out:
        out.release()
    pbar.close()
    print(f"Processed {frame_count} frames. Output saved to {output_path}")

# ----------------------
# 4. Webcam Demo (real-time)
# ----------------------
def webcam_demo(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleEmotionCNN(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay, pred_idx = overlay_saliency_on_frame(model, frame, device)
        label = IDX_TO_EMOTION[pred_idx]
        cv2.putText(overlay, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('Webcam Emotion Recognition', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ----------------------
# CLI Entrypoints
# ----------------------
if __name__ == '__main__':
    import argparse
    import sys
    
    MODEL_PATH_DEFAULT = 'C:/Users/keysc/Desktop/EmotionCNN/emotion_model.pth'
    TRAIN_DIR_DEFAULT = 'C:/Users/keysc/Desktop/EmotionCNN/archive/train'
    TEST_DIR_DEFAULT = 'C:/Users/keysc/Desktop/EmotionCNN/archive/test'
    
    # Check if running in IDE (interactive mode) or command line
    if len(sys.argv) == 1:
        # Interactive mode for IDE
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
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == '1':
                    print("\n=== TRAINING MODEL ===")
                    train_dir = input(f"Training directory (default: {TRAIN_DIR_DEFAULT}): ").strip()
                    if not train_dir:
                        train_dir = TRAIN_DIR_DEFAULT
                    
                    val_dir = input(f"Validation directory (default: {TEST_DIR_DEFAULT}): ").strip()
                    if not val_dir:
                        val_dir = TEST_DIR_DEFAULT
                    
                    epochs = input("Number of epochs (default: 30): ").strip()
                    epochs = int(epochs) if epochs else 30
                    
                    batch_size = input("Batch size (default: 64): ").strip()
                    batch_size = int(batch_size) if batch_size else 64
                    
                    lr = input("Learning rate (default: 0.001): ").strip()
                    lr = float(lr) if lr else 0.001
                    
                    save_path = input(f"Model save path (default: {MODEL_PATH_DEFAULT}): ").strip()
                    if not save_path:
                        save_path = MODEL_PATH_DEFAULT
                    
                    print(f"\nStarting training with:")
                    print(f"Train dir: {train_dir}")
                    print(f"Val dir: {val_dir}")
                    print(f"Epochs: {epochs}")
                    print(f"Batch size: {batch_size}")
                    print(f"Learning rate: {lr}")
                    print(f"Save path: {save_path}")
                    
                    train_from_scratch(train_dir, val_dir, epochs, batch_size, lr, save_path)
                    
                elif choice == '2':
                    print("\n=== BATCH CLASSIFICATION ===")
                    model_path = input(f"Model path (default: {MODEL_PATH_DEFAULT}): ").strip()
                    if not model_path:
                        model_path = MODEL_PATH_DEFAULT
                    folder_path = input(f"Folder path to classify (default: {TEST_DIR_DEFAULT}): ").strip()
                    if not folder_path:
                        folder_path = TEST_DIR_DEFAULT
                    output_csv = input("Output CSV filename (default: results.csv): ").strip()
                    if not output_csv:
                        output_csv = 'results.csv'
                    
                    classify_folder_to_csv(model_path, folder_path, output_csv)
                    
                elif choice == '3':
                    print("\n=== VIDEO PROCESSING ===")
                    model_path = input(f"Model path (default: {MODEL_PATH_DEFAULT}): ").strip()
                    if not model_path:
                        model_path = MODEL_PATH_DEFAULT
                    video_path = input("Input video path: ").strip()
                    output_path = input("Output video path (default: output_video.avi): ").strip()
                    if not output_path:
                        output_path = 'output_video.avi'
                    
                    process_video(model_path, video_path, output_path)
                    
                elif choice == '4':
                    print("\n=== WEBCAM DEMO ===")
                    model_path = input(f"Model path (default: {MODEL_PATH_DEFAULT}): ").strip()
                    if not model_path:
                        model_path = MODEL_PATH_DEFAULT
                    print("Starting webcam demo... Press 'q' to quit.")
                    webcam_demo(model_path)
                    
                elif choice == '5':
                    print("Exiting...")
                    break
                    
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
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
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")
    
    else:
        # Command line mode
        parser = argparse.ArgumentParser(description='Emotion CNN - Train, Batch Classify, Video Demo, Webcam Demo')
        subparsers = parser.add_subparsers(dest='command')

        # Train
        train_parser = subparsers.add_parser('train', help='Train model from scratch')
        train_parser.add_argument('--train_dir', default=TRAIN_DIR_DEFAULT)
        train_parser.add_argument('--val_dir', default=TEST_DIR_DEFAULT)
        train_parser.add_argument('--epochs', type=int, default=30)
        train_parser.add_argument('--batch_size', type=int, default=64)
        train_parser.add_argument('--lr', type=float, default=0.001)
        train_parser.add_argument('--save_path', default=MODEL_PATH_DEFAULT)

        # Batch classify
        classify_parser = subparsers.add_parser('classify', help='Batch classify images in a folder and output CSV')
        classify_parser.add_argument('--model_path', default=MODEL_PATH_DEFAULT)
        classify_parser.add_argument('--folder_path', default=TEST_DIR_DEFAULT)
        classify_parser.add_argument('--output_csv', default='results.csv')

        # Video demo
        video_parser = subparsers.add_parser('video', help='Process a video, classify, overlay saliency, save video')
        video_parser.add_argument('--model_path', default=MODEL_PATH_DEFAULT, required=False)
        video_parser.add_argument('--video_path', required=True)
        video_parser.add_argument('--output_path', default='output_video.avi')

        # Webcam demo
        webcam_parser = subparsers.add_parser('webcam', help='Webcam real-time demo')
        webcam_parser.add_argument('--model_path', default=MODEL_PATH_DEFAULT, required=False)

        args = parser.parse_args()
        if args.command == 'train':
            train_from_scratch(args.train_dir, args.val_dir, args.epochs, args.batch_size, args.lr, args.save_path)
        elif args.command == 'classify':
            classify_folder_to_csv(args.model_path, args.folder_path, args.output_csv)
        elif args.command == 'video':
            process_video(args.model_path, args.video_path, args.output_path)
        elif args.command == 'webcam':
            webcam_demo(args.model_path)
        else:
            parser.print_help() 