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