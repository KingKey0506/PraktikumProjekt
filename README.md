# EmotionCNN - Emotion Recognition with Convolutional Neural Networks

Ein Deep Learning Projekt zur Erkennung von Emotionen in Gesichtern mit Hilfe von Convolutional Neural Networks (CNN).

## Projektbeschreibung

Dieses Projekt implementiert ein CNN-basiertes System zur Erkennung von 6 verschiedenen Emotionen:
- Happy (Glücklich)
- Surprise (Überrascht)
- Sad (Traurig)
- Angry (Wütend)
- Disgust (Ekel)
- Fear (Angst)

## Technologien

- **Python 3.x**
- **PyTorch** - Deep Learning Framework
- **OpenCV** - Bildverarbeitung
- **dlib** - Gesichtserkennung
- **scikit-learn** - Metriken und Evaluation
- **matplotlib** - Visualisierung
- **tqdm** - Fortschrittsanzeige

## Projektstruktur

```
EmotionCNN/
├── archive/           # Dataset (train/test Ordner)
├── praktikum/         # Hauptcode
│   └── main.py       # Hauptskript
├── .gitignore        # Git Ignore Datei
└── README.md         # Diese Datei
```

## Installation

1. Repository klonen:
```bash
git clone https://github.com/KingKey0506/PraktikumProjekt.git
cd PraktikumProjekt
```

2. Abhängigkeiten installieren:
```bash
pip install torch torchvision
pip install opencv-python
pip install dlib
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install imutils
```

## Verwendung

1. Dataset vorbereiten:
   - Lege deine Bilder in die entsprechenden Ordner unter `archive/train/` und `archive/test/`
   - Jede Emotion sollte einen eigenen Ordner haben (happy, surprise, sad, angry, disgust, fear)

2. Training starten:
```bash
cd praktikum
python main.py
```

3. Modell verwenden:
```python
from main import EmotionRecognizer

# Modell laden
recognizer = EmotionRecognizer("path/to/train/dir")
recognizer.load_model()

# Emotion vorhersagen
emotion, confidence = recognizer.predict_emotion("path/to/image.jpg")
```

## Features

- **Transfer Learning**: Verwendet pre-trained ResNet18 als Backbone
- **Data Augmentation**: Automatische Bildverarbeitung für bessere Generalisierung
- **Face Detection**: Automatische Gesichtserkennung und -ausschnitt
- **Early Stopping**: Verhindert Overfitting
- **Class Weighting**: Behandlung von unausgewogenen Datasets
- **Caching**: Vorverarbeitete Bilder werden zwischengespeichert

## Modellarchitektur

- **Backbone**: ResNet18 (pre-trained auf ImageNet)
- **Fine-tuning**: Nur die letzten 30 Layer werden trainiert
- **Classifier**: Custom Head mit Dropout und BatchNorm
- **Input**: 64x64 RGB Bilder
- **Output**: 6 Emotion-Klassen

## Metriken

Das Modell wird mit folgenden Metriken evaluiert:
- Accuracy
- Precision, Recall, F1-Score pro Emotion
- Confusion Matrix

## Lizenz

Dieses Projekt ist Teil eines Praktikums. 


# FER2013 Emotion Recognition (ResNet18/ResNet50)

A PyTorch project for facial expression recognition on FER2013. It provides custom ResNet18/ResNet50 backbones, configurable augmentation, warmup + LR schedulers, TensorBoard logging, checkpointing, and an optional webcam demo with saliency overlay.

## Features
- ResNet18/ResNet50 heads for 7 emotion classes
- AdaptiveAvgPool2d((1,1)) head (supports variable input sizes; match training size for best accuracy)
- Data augmentation: RandomResizedCrop, HorizontalFlip, Rotation
- SGD optimizer with momentum and weight decay (AdamW available)
- LR warmup + schedulers: CosineAnnealingLR or ReduceLROnPlateau
- TensorBoard logging, best-checkpoint saving
- Webcam demo with OpenCV DNN face detector, saliency visualization

## Model architectures
Number of trainable parameters in net50: 23516167
Number of trainable parameters in net18: 11173383
- ResNet18 head
  - Conv-BN-ReLU + MaxPool → 4 residual stages → AdaptiveAvgPool2d((1,1)) → Flatten → Linear(512→7)
- ResNet50 head
  - Conv-BN-ReLU + MaxPool → 4 bottleneck stages → AdaptiveAvgPool2d((1,1)) → Flatten → Dropout(0.5) → Linear(2048→7)

Note: While AdaptiveAvgPool2d allows variable inputs, inference should use the same input size used in training (e.g., 48×48 or 224×224) to avoid accuracy drop and excessive GPU memory usage.

## Data
- Dataset: FER2013 CSV (pixels are space-separated grayscale values)
- Default path: resnet/fer2013.csv
- Pass a custom path via CLI: --data-path path/to/fer2013.csv

## Key files
- train_cv.py
  - Model definitions: net18, net50
  - Training loop: train(...)
  - LR schedulers: warmup (LinearLR) + CosineAnnealingLR or ReduceLROnPlateau
  - Evaluation helper: eaccuracy_gpu(...)
  - Custom dataset class (fersets)
- test.py
  - Local CLI entry for training (argparse)
  - Loads FER2013 CSV, splits/builds DataLoaders, starts training
  - Writes TensorBoard logs, saves best checkpoint
- result.py
  - WebcamDemo: real-time inference with face detection
  - overlay_saliency_on_frame: saliency overlay on frames/ROIs
  - batch_attention_vis and evaluation utilities (confusion matrix, metrics)
- remote/fabfile.py
  - Fabric task run_train for remote training
- runs/ (generated)
  - TensorBoard logs per run
- weights/ (generated)
  - Best model weights per run

## Installation
- Python 3.9+ recommended
- Install dependencies:
  - PyTorch + torchvision (CUDA optional)
  - numpy, pandas, scikit-learn, tensorboard, opencv-python, pillow, tqdm
## Tipps
- For 4GB GPUs, start with --reshape 48 and --batch-size 8–32.
- Logs: runs/-net{NET}-lr{LR}-{TIMESTAMP}
- Weights: weights/-net{NET}-lr{LR}-{TIMESTAMP}/model_{NET}_lr{LR}.pth
- Best weights are saved when test accuracy improves.
## Remote training with Fabric
Example (adjust host in remote/fabfile.py):
  To train the model using Fabric, run:
````bash
fab run_train:epochs=50,batch_size=64,lr=0.03,net="net50"
````
##View logs:
````bash
tensorboard --logdir runs
````
Example (Windows, conda env):
````bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn tensorboard opencv-python pillow tqdm
````
