import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from PIL import Image
import dlib
from imutils import face_utils
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import models

# Emotion labels
EMOTIONS = ['happy', 'surprise', 'sad', 'angry', 'disgust', 'fear']
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}


class FacialImageDataset(Dataset):
    """Dataset class for facial emotion recognition using CNN on raw images"""

    def __init__(self, data_dir, transform=None, precompute=True):
        """
        Initialize dataset
        Args:
            data_dir: Directory containing emotion folders
            transform: Image transformations
            precompute: If True, precompute all processed images
        """
        self.data_dir = data_dir
        self.samples = []
        self.labels = []
        self.transform = transform
        self.precompute = precompute

        # Speicherpfade für verarbeitete Bilder
        self.image_path = os.path.join(self.data_dir, 'processed_images.npy')
        self.label_path = os.path.join(self.data_dir, 'labels.npy')

        # Wenn Dateien existieren, lade sie, sonst berechne und speichere sie
        if self.precompute and os.path.exists(self.image_path) and os.path.exists(self.label_path):
            # Load existing processed images
            print("Lade verarbeitete Bilder aus Cache...")
            self.samples = np.load(self.image_path)
            self.labels = np.load(self.label_path)
            print(f"Geladen: {len(self.samples)} verarbeitete Bilder")
        else:
            # Calculate and save processed images
            print("Verarbeite alle Bilder und speichere sie...")
            self._load_dataset()
            if self.precompute:
                np.save(self.image_path, np.array(self.samples))
                np.save(self.label_path, np.array(self.labels))
                print(f"Verarbeitete Bilder gespeichert unter {self.image_path}")

    def _load_dataset(self):
        """Load images and preprocess them"""
        if self.precompute:
            for emotion in EMOTIONS:
                emotion_dir = os.path.join(self.data_dir, emotion)
                if not os.path.exists(emotion_dir):
                    print(f"Warning: {emotion_dir} not found, skipping...")
                    continue

                print(f"Processing {emotion} images...")
                img_files = [f for f in os.listdir(emotion_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for img_file in tqdm(img_files, desc=f"Loading {emotion}"):
                    img_path = os.path.join(emotion_dir, img_file)
                    processed_img = self._process_image(img_path)
                    if processed_img is not None:
                        self.samples.append(processed_img)
                        self.labels.append(EMOTION_TO_IDX[emotion])

            print(f"Loaded {len(self.samples)} samples")
        else:
            # Store image paths for on-the-fly processing
            self.image_paths = []
            for emotion in EMOTIONS:
                emotion_dir = os.path.join(self.data_dir, emotion)
                if not os.path.exists(emotion_dir):
                    print(f"Warning: {emotion_dir} not found, skipping...")
                    continue

                img_files = [f for f in os.listdir(emotion_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for img_file in img_files:
                    img_path = os.path.join(emotion_dir, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(EMOTION_TO_IDX[emotion])

            print(f"Found {len(self.image_paths)} images")

    def _process_image(self, img_path):
        """Process image for CNN input"""
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect and crop face
            face_img = self._detect_and_crop_face(img)
            
            # Resize to 64x64 for faster training
            face_img = cv2.resize(face_img, (64, 64))
            
            # Normalize to [0, 1]
            face_img = face_img.astype(np.float32) / 255.0
            
            # Convert to channel-first format (C, H, W)
            face_img = np.transpose(face_img, (2, 0, 1))
            
            return face_img
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

    def _detect_and_crop_face(self, img):
        """Detect and crop face from image"""
        try:
            detector = dlib.get_frontal_face_detector()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = detector(gray, 1)
            
            if len(faces) == 0:
                # Fallback: use the entire image if no face detected
                return img
            
            # Get the largest face
            face = max(faces, key=lambda rect: rect.area())
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            
            # Add some margin
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img.shape[1], x2 + margin)
            y2 = min(img.shape[0], y2 + margin)
            
            return img[y1:y2, x1:x2]
            
        except Exception as e:
            # Fallback: use the entire image if detection fails
            print(f"Face detection failed, using entire image: {e}")
            return img

    def __len__(self):
        if self.precompute:
            return len(self.samples)
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):
        if self.precompute:
            img = self.samples[idx]
            
            # Convert from numpy array (C, H, W) to PIL Image
            if isinstance(img, np.ndarray):
                # img is in format (C, H, W), convert to (H, W, C) for PIL
                img = np.transpose(img, (1, 2, 0))
                # Denormalize from [0,1] to [0,255] for PIL
                img = (img * 255).astype(np.uint8)
            
            if self.transform:
                img = self.transform(img)
            else:
                # Convert back to tensor if no transform
                img = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
                
            return img, torch.LongTensor([self.labels[idx]])
        else:
            # Process on-the-fly
            img_path = self.image_paths[idx]
            img = self._process_image(img_path)
            if img is None:
                # Return zeros if processing fails
                img = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                # Convert from (C, H, W) to (H, W, C) for transformations
                img = np.transpose(img, (1, 2, 0))
                img = (img * 255).astype(np.uint8)
                
            if self.transform:
                img = self.transform(img)
            else:
                # Convert to tensor
                img = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
                
            return img, torch.LongTensor([self.labels[idx]])


class EmotionCNN(nn.Module):
    """CNN for emotion recognition using transfer learning"""

    def __init__(self, num_classes=6, dropout=0.3):
        super(EmotionCNN, self).__init__()
        
        # Load pre-trained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze early layers for transfer learning (freeze more layers for better transfer)
        for param in list(self.backbone.parameters())[:-30]:  # Freeze all but last 30 layers
            param.requires_grad = False
        
        # Replace the final layer with optimized architecture
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout in final layers
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),  # Minimal dropout before output
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class EmotionRecognizer:
    """Main class for training and inference"""

    def __init__(self, train_dir, test_dir=None, model_save_path="emotion_model.pth"):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model with optimized dropout
        self.model = EmotionCNN(dropout=0.3).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimized optimizer settings for transfer learning
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.0003,  # Slightly higher LR for better convergence
            weight_decay=0.01,  # Stronger regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Optimized scheduler for better learning rate adaptation
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=8,  # More responsive
            factor=0.3,  # More aggressive reduction
            min_lr=1e-7  # Minimum learning rate
        )

        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def load_data(self, batch_size=64, split_ratio=0.7):
        """Load and split dataset with data augmentation"""
        print("Loading training dataset...")
        
        # Enhanced data augmentation for training
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Slightly more rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),  # More aggressive cropping
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Simple transform for validation
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load training dataset
        full_dataset = FacialImageDataset(self.train_dir, transform=None, precompute=True)

        if len(full_dataset) == 0:
            print("No training data loaded! Please check your dataset directory structure.")
            return None, None

        # Split training dataset into train/validation with better ratio
        train_size = int(split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_dataset)), [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create separate datasets with different transforms
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)
        
        # Apply transforms
        full_dataset.transform = train_transform  # This will affect train_dataset
        
        # Create validation dataset with different transform
        val_full_dataset = FacialImageDataset(self.train_dir, transform=val_transform, precompute=True)
        val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices.indices)

        # Create data loaders with optimized settings
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        print(f"Training samples: {train_size}, Validation samples: {val_size}")

        # Calculate class weights for imbalanced dataset with smoothing
        all_labels = []
        for i in train_indices.indices:
            all_labels.append(full_dataset.labels[i])
        
        class_counts = np.bincount(all_labels)
        # Add smoothing to prevent extreme weights
        class_counts = class_counts + 1  # Add 1 to each class count
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        return train_loader, val_loader

    def load_test_data(self, batch_size=64):
        """Load test dataset for final evaluation"""
        if self.test_dir is None:
            print("No test directory provided!")
            return None
            
        print("Loading test dataset...")
        
        # Simple transform for test data
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load test dataset
        test_dataset = FacialImageDataset(self.test_dir, transform=test_transform, precompute=True)
        
        if len(test_dataset) == 0:
            print("No test data loaded!")
            return None
            
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        print(f"Test samples: {len(test_dataset)}")
        
        return test_loader

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training", mininterval=1.0)
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.squeeze().to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating", mininterval=1.0):
                data, target = data.to(self.device), target.squeeze().to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy, all_predictions, all_targets

    def test_model(self, test_loader):
        """Test the model on separate test dataset"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing", mininterval=1.0):
                data, target = data.to(self.device), target.squeeze().to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy, all_predictions, all_targets

    def train(self, epochs=150, early_stopping_patience=15):
        """Train the model"""
        train_loader, val_loader = self.load_data()
        if train_loader is None:
            return

        best_accuracy = 0
        patience_counter = 0

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Early stopping patience: {early_stopping_patience}")
        print("=" * 60)

        for epoch in range(epochs):
            print(f'\nEpoch [{epoch + 1}/{epochs}]')

            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # Early stopping and model saving
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                patience_counter = 0
                self.save_model()
                print(f"✓ New best accuracy: {best_accuracy:.2f}%")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{early_stopping_patience})")

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Print final validation results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
        print("\nValidation Classification Report:")
        print(classification_report(val_targets, val_preds, target_names=EMOTIONS, digits=4))

        # Test on separate test dataset
        if self.test_dir:
            print("\n" + "=" * 60)
            print("TESTING ON SEPARATE TEST DATASET")
            print("=" * 60)
            test_loader = self.load_test_data()
            if test_loader:
                test_loss, test_acc, test_preds, test_targets = self.test_model(test_loader)
                print(f"Test Accuracy: {test_acc:.2f}%")
                print(f"Test Loss: {test_loss:.4f}")
                print("\nTest Classification Report:")
                print(classification_report(test_targets, test_preds, target_names=EMOTIONS, digits=4))

    def save_model(self):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, self.model_save_path)

    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.model_save_path):
            checkpoint = torch.load(self.model_save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Model loaded successfully!")
            return True
        else:
            print(f"No saved model found at {self.model_save_path}")
            return False

    def predict_emotion(self, img_path):
        """Predict emotion for a single image"""
        self.model.eval()

        # Process image
        img = self._process_image_static(img_path)
        if img is None:
            return "Could not process image", 0.0

        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(img).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()

        return EMOTIONS[predicted_idx], confidence

    def _process_image_static(self, img_path):
        """Static method for image processing"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_img = self._detect_and_crop_face_static(img)
            if face_img is None:
                return None
            
            face_img = cv2.resize(face_img, (64, 64))
            face_img = face_img.astype(np.float32) / 255.0
            face_img = np.transpose(face_img, (2, 0, 1))
            
            # Normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            face_img = (face_img - mean[:, None, None]) / std[:, None, None]
            
            return face_img
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

    def _detect_and_crop_face_static(self, img):
        """Static method for face detection and cropping"""
        try:
            detector = dlib.get_frontal_face_detector()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = detector(gray, 1)
            
            if len(faces) == 0:
                # Fallback: use the entire image if no face detected
                return img
            
            face = max(faces, key=lambda rect: rect.area())
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img.shape[1], x2 + margin)
            y2 = min(img.shape[0], y2 + margin)
            
            return img[y1:y2, x1:x2]
            
        except Exception as e:
            # Fallback: use the entire image if detection fails
            return img

    def predict_batch(self, img_paths):
        """Predict emotions for multiple images"""
        self.model.eval()
        results = []

        for img_path in tqdm(img_paths, desc="Predicting"):
            emotion, confidence = self.predict_emotion(img_path)
            results.append({
                'image': img_path,
                'emotion': emotion,
                'confidence': confidence
            })

        return results

    def plot_training_history(self):
        """Plot training history"""
        if not self.train_losses:
            print("No training history to plot")
            return

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Val Loss', alpha=0.8)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', alpha=0.8)
        plt.plot(self.val_accuracies, label='Val Accuracy', alpha=0.8)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(np.array(self.val_accuracies) - np.array(self.train_accuracies), alpha=0.8)
        plt.title('Accuracy Gap (Val - Train)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Difference (%)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Usage example and utility functions
def create_sample_dataset_structure(base_path):
    """Create sample dataset directory structure"""
    os.makedirs(base_path, exist_ok=True)
    for emotion in EMOTIONS:
        emotion_path = os.path.join(base_path, emotion)
        os.makedirs(emotion_path, exist_ok=True)
    print(f"Created dataset structure at: {base_path}")
    print("Please add your emotion images to the respective folders.")


def analyze_dataset(data_dir):
    """Analyze dataset distribution"""
    print("Dataset Analysis:")
    print("=" * 40)
    total_images = 0

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(data_dir, emotion)
        if os.path.exists(emotion_dir):
            img_count = len([f for f in os.listdir(emotion_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{emotion.capitalize()}: {img_count} images")
            total_images += img_count
        else:
            print(f"{emotion.capitalize()}: Directory not found")

    print(f"\nTotal images: {total_images}")
    if total_images > 0:
        print(f"Average per emotion: {total_images / len(EMOTIONS):.1f}")


if __name__ == "__main__":
    # Set your dataset directories here
    TRAIN_DIR = "C:/Users/keysc/Desktop/EmotionCNN/archive/train"
    TEST_DIR = "C:/Users/keysc/Desktop/EmotionCNN/archive/test"

    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    # Analyze datasets
    print("\n" + "=" * 60)
    print("TRAINING DATASET ANALYSIS")
    print("=" * 60)
    if os.path.exists(TRAIN_DIR):
        analyze_dataset(TRAIN_DIR)
    else:
        print(f"Training dataset directory not found: {TRAIN_DIR}")
        exit()

    print("\n" + "=" * 60)
    print("TEST DATASET ANALYSIS")
    print("=" * 60)
    if os.path.exists(TEST_DIR):
        analyze_dataset(TEST_DIR)
    else:
        print(f"Test dataset directory not found: {TEST_DIR}")
        TEST_DIR = None  # Set to None if test directory doesn't exist

    # Initialize recognizer with both datasets
    recognizer = EmotionRecognizer(TRAIN_DIR, TEST_DIR)

    # Train the model
    print("\nTraining the emotion recognition model...")
    recognizer.train(epochs=100)

    # Plot training history
    recognizer.plot_training_history()

    print("\nTraining completed! Model saved as 'emotion_model.pth'")
    print("\nTo use the trained model:")
    print("1. Load the model: recognizer.load_model()")
    print("2. Single prediction: emotion, confidence = recognizer.predict_emotion('image_path')")
    print("3. Batch prediction: results = recognizer.predict_batch(['img1.jpg', 'img2.jpg'])")