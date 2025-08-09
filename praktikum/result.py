import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import random
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from train_cv import net50  
from train_cv import net18

def batch_attention_vis(weight_path , csv_path , device='cuda', n_samples=12):

    net = net50
    model = net.to(device)
    
    # Handle relative paths - make them relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  
    
    if not os.path.isabs(weight_path):
        # Try script directory first
        candidate_path = os.path.join(script_dir, weight_path)
        if os.path.exists(candidate_path):
            weight_path = candidate_path
        else:
            # Try project root directory
            candidate_path = os.path.join(project_root, weight_path)
            if os.path.exists(candidate_path):
                weight_path = candidate_path
            else:
                # Keep original path 
                weight_path = os.path.join(script_dir, weight_path)
    
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(script_dir, csv_path)
    
    print(f"Looking for weight at: {weight_path}")
    print(f"Weight exists: {os.path.exists(weight_path)}")
    print(f"Looking CSV  at: {csv_path}")
    print(f"CSV exists: {os.path.exists(csv_path)}")
    
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    df = pd.read_csv(csv_path)
    label_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

  
    samples = []
    for label in range(len(label_names)):
        subset = df[df['emotion'] == label]
        if len(subset) == 0:
            print(f'Label {label_names[label]} has no samples, skipping.')
            continue
        row = subset.sample(n=1).iloc[0]
        samples.append((row, label))

   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, n_samples * 2.5))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, (row, label) in enumerate(samples):
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, mode='L').resize((224, 224))
        label_str = label_names[label]
        input_tensor = transform(img).unsqueeze(0).to(device)



        
        features = []
        def hook_fn(module, input, output):
            features.append(output.cpu().data.numpy())
        last_block = model[4][-1]
        conv = last_block.conv3
        handle = conv.register_forward_hook(hook_fn)

        
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
        handle.remove()
        feature_map = features[0][0]  # [C, H, W]

        # Grad-CAM
        params = list(model[-1].parameters())
        weight_softmax = params[0].cpu().data.numpy()
        class_weights = weight_softmax[pred]
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        for i, w in enumerate(class_weights):
            cam += w * feature_map[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if np.max(cam) > 0:
            cam /= np.max(cam)

        heatmap = cam  # [224,224], float
        img_np = np.array(img).astype(np.float32) / 255  # [224,224], float
        overlay = 0.5 * heatmap + 0.5 * img_np

        
        ax1, ax2, ax3 = axes[idx]
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title(f'Label: {label_str}',fontsize=7)
        ax1.axis('off')

        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title('Grad-CAM')
        ax2.axis('off')

        ax3.imshow(overlay, cmap='jet')
        ax3.set_title(f'Pred: {label_names[pred]}',fontsize=7)
        ax3.axis('off')

    plt.tight_layout()
    plt.show()


# batch_attention_vis(
#     'weights/model_net50_lr0.025.pth',
#     'resnet/fer2013.csv',
#     device='cuda',
#     n_samples=7
# )
def count_parameters(net50):
    return sum(p.numel() for p in net50.parameters() if p.requires_grad)
print(f"Number of trainable parameters in net50: {count_parameters(net50)}")

print(f"Number of trainable parameters in net18: {count_parameters(net18)}")

def overlay_saliency_on_frame(model, frame, device, face_roi=None):
    if face_roi is not None:
        x1, y1, x2, y2 = face_roi
        roi = frame[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
        roi = frame
    #print("roi.shape:", roi.shape)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (64, 64))  
    #print("img_resized.shape:", img_resized.shape) 
    transform = transforms.Compose([
        transforms.ToTensor(),
        
        # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    
    img_pil = Image.fromarray(img_resized, mode='L')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    #print("input_tensor.shape:", input_tensor.shape)  

    # Forward + backward
    output = model(input_tensor)
    pred_idx = int(output.argmax(dim=1))
    model.zero_grad()
    output[0, pred_idx].backward()

    grad = input_tensor.grad.abs().squeeze().cpu().numpy()
    #print("grad.shape:", grad.shape)  

    saliency = np.max(grad, axis=0)
    #print("saliency.shape:", saliency.shape)  

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = np.uint8(255 * saliency)
    saliency = cv2.resize(saliency, (x2-x1, y2-y1))

    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    #print("heatmap.shape:", heatmap.shape) 

    overlay = frame.copy()
    #print("frame[y1:y2, x1:x2].shape:", frame[y1:y2, x1:x2].shape)
    overlay[y1:y2, x1:x2] = cv2.addWeighted(
        frame[y1:y2, x1:x2], 0.7,
        heatmap,            0.3,
        0
    )
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    max_prob = float(probs[pred_idx])
    
    return overlay, pred_idx, max_prob
emotions = ['Anger','Disgust','Fear','Happy','Neutral','Sadness','Surprise']
EmotionToIndex = {emotion: idx for idx, emotion in enumerate(emotions)}
IndexToEmotion = {idx: emotion for emotion, idx in EmotionToIndex.items()}
def WebcamDemo(ModelPath, network_type=None):
    """
    Webcam demo for emotion recognition
    Args:
        ModelPath: Path to the trained model
        network_type: 'net50' or 'net18', if None will try to infer from path
    """
    caffemodel = r"E:\githb\PraktikumProjekt\praktikum\res10_300x300_ssd_iter_140000.caffemodel"
    prototxt = r"E:\githb\PraktikumProjekt\praktikum\deploy.prototxt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect network type from model path if not specified
    if network_type is None:
        if 'net18' in ModelPath:
            network_type = 'net18'
        else:
            network_type = 'net50' 
    
    # Load model
    if network_type == 'net18':
        model = net18
        print(f"Using ResNet18 for webcam demo")
    else:
        model = net50
        print(f"Using ResNet50 for webcam demo")
    
    print(f"Loading model from: {ModelPath}")
    print(f"Model exists: {os.path.exists(ModelPath)}")
    
    try:
        model.load_state_dict(torch.load(ModelPath, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test webcam access
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit, 's' to save screenshot")
    print("Proto exists:", os.path.exists(prototxt))
    print("Caffe model exists:", os.path.exists(caffemodel))
    
    # Load face detection model
    if os.path.exists(prototxt) and os.path.exists(caffemodel):
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        face_detection_enabled = True
        print("Face detection enabled")
    else:
        face_detection_enabled = False
        print("Face detection disabled (model files not found)")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Display input format info every 30 frames
        if frame_count % 30 == 1:
            print(f"Input frame format: {frame.shape}, dtype: {frame.dtype}")
        
        # Show raw frame in separate window
        cv2.imshow("Raw Input", frame)
        
        # Face detection and emotion recognition
        best_box = None
        if face_detection_enabled:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            best_conf = 0
            for i in range(detections.shape[2]):
                conf = float(detections[0,0,i,2])
                if conf > 0.6 and conf > best_conf:
                    best_conf = conf
                    box = detections[0,0,i,3:7] * np.array([w, h, w, h])
                    best_box = box.astype(int)

        # Emotion recognition
        if best_box is not None:
            x1,y1,x2,y2 = best_box
            overlay, pred_idx, max_prob = overlay_saliency_on_frame(
                model, frame, device, face_roi=(x1,y1,x2,y2)  
            )
            # Draw face bounding box
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
            status_text = f"Face detected, confidence: {max_prob:.2f}"
        else:
            overlay, pred_idx, max_prob = overlay_saliency_on_frame(model, frame, device)
            status_text = "No face detected, using full frame"
        
        label = IndexToEmotion[pred_idx]
        
        # Add text overlay
        cv2.putText(overlay, f'Emotion: {label} ({max_prob:.2f})', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(overlay, status_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(overlay, f'Network: {network_type}', (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        cv2.imshow('Webcam Emotion Recognition', overlay)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f'webcam_screenshot_{frame_count}.jpg'
            cv2.imwrite(screenshot_path, overlay)
            print(f"Screenshot saved: {screenshot_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam demo ended")

# WebcamDemo(r"weights/model_net50_lr0.025.pth") 

def evaluate_confusion_matrix(weight_path, csv_path, network_type='net50', device='cuda:0', save_path=None, max_samples=None):
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import seaborn as sns
    
    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if not os.path.isabs(weight_path):
        candidate_path = os.path.join(script_dir, weight_path)
        if os.path.exists(candidate_path):
            weight_path = candidate_path
        else:
            candidate_path = os.path.join(project_root, weight_path)
            if os.path.exists(candidate_path):
                weight_path = candidate_path
            else:
                weight_path = os.path.join(script_dir, weight_path)
    
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(script_dir, csv_path)
    
    print(f"Loading model from: {weight_path}")
    print(f"Loading data from: {csv_path}")
    print(f"Using network: {network_type}")
    print(f"Using device: {device}")
    
    # Load appropriate model
    if network_type == 'net18':
        model = net18
    else:
        model = net50
    
    model = model.to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load  data
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"Dataset contains {total_samples} samples")
    
    # Limit samples if specified
    if max_samples is not None and max_samples < total_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Randomly sampling {max_samples} samples for evaluation")
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Prepare data transformation - both networks expect grayscale input
    if network_type == 'net50':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:  # net18
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    true_labels = []
    predicted_labels = []
    print("Evaluating model on test data...")
    
    for idx, row in df.iterrows():
        #  pixel data
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, mode='L')  # Keep grayscale
        img_tensor = transform(img).unsqueeze(0).to(device)
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
        
        true_labels.append(row['emotion'])
        predicted_labels.append(predicted_label)
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} samples...")
    
    print(f"Evaluation complete! Processed {len(true_labels)} samples.")
    
    #metrics
    cm = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    #perclass accuracies
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    #results
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nPer-class Accuracies:")
    for i, acc in enumerate(class_accuracies):
        print(f"{emotion_labels[i]}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Print  classification report
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=emotion_labels))
    
    # visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                cbar_kws={'label': 'Number of Samples'})
    plt.title(f'Confusion Matrix - {network_type.upper()}\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add percentage annotations
    for i in range(len(emotion_labels)):
        for j in range(len(emotion_labels)):
            percentage = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            plt.text(j+0.5, i+0.7, f'{percentage:.1f}%', 
                    ha='center', va='center', fontsize=8, color='red')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, accuracy, class_accuracies

def evaluate_model_performance(weight_path, csv_path, network_type='net50', device='cuda'):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    cm, accuracy, class_accuracies = evaluate_confusion_matrix(
        weight_path, csv_path, network_type, device
    )
    
    # Calculate additional metrics
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    results = {
        'confusion_matrix': cm,
        'overall_accuracy': accuracy,
        'class_accuracies': dict(zip(emotion_labels, class_accuracies)),
        'network_type': network_type,
        'total_samples': cm.sum()
    }
    return results
# cm, accuracy, class_acc = evaluate_confusion_matrix(
#     weight_path='weights/model_net50_lr0.025.pth',
#     csv_path='resnet/fer2013.csv',
#     network_type='net50',
#     device='cuda',
#     save_path='confusion_matrix_net50.png',
#     max_samples=None
# )

