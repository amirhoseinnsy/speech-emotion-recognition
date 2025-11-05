# Speech Emotion Recognition (Assignment 5)

This repository contains my implementation for **Assignment 5** of the *Deep Learning for Audio and Image Applications* course at the University of Tehran.  
The project focuses on **Speech Emotion Recognition (SER)** using both traditional and modern feature extraction techniques (Log-Mel spectrograms and HuBERT embeddings).

---

## 📂 Project Structure

```bash
ASSIGNMENT5/
├── config/
│ ├── config.yaml # General configuration
│ ├── log_hubert.yaml # HuBERT feature extraction setup
│ └── log_mel.yaml # Mel-spectrogram setup
│
├── data/
│ ├── init.py
│ ├── data_loader.py # Handles dataset loading and preprocessing
│ └── dataset/ # Raw or processed data (CREMA-D)
│
├── models/
│ ├── model.py # Neural network architectures (CNN / MLP)
│ └── saved_models/ # Directory for trained models
│
├── scripts/
│ ├── main.py # Entry point for running the whole pipeline
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation and testing
│ └── optim.py # Optimization utilities
│
├── utils/
│ ├── features.py # Feature extraction (Log-Mel & HuBERT)
│ ├── metrics.py # Accuracy, confusion matrix, etc.
│ ├── visualization.py # Plot loss curves, confusion matrix, etc.
│ └── init.py
│
├── picture/ # Saved plots and visualizations
├── debug_log.txt
├── report.pdf # Written report of the assignment
└── HWExtra.pdf / Report_template.docx
```

---

## 🧠 Task Description

The goal is to build and compare two pipelines for classifying emotions from speech audio:

1. **Traditional approach:**  
   Extract **Log-Mel spectrogram** features using `librosa` or `torchaudio`, then train a CNN-based classifier.

2. **Modern approach (Self-Supervised):**  
   Extract **HuBERT** embeddings from the pre-trained `facebook/hubert-base-ls960` model and train an MLP classifier.

Dataset used: **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**  
Classes: *Neutral, Happy, Sad, Angry* (subset for efficiency)

---

## ⚙️ Usage

### 1. Setup
```bash
git clone https://github.com/<amirhoseinnsy>/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
```

2. Configuration

Edit parameters in config/config.yaml (dataset path, batch size, learning rate, etc.).
3. Run training

```bash
python scripts/train.py
```

4. Evaluate model

```bash
python scripts/evaluate.py
```

5. Visualize results

Plots are saved under /picture and include training loss, accuracy, and confusion matrix.
📊 Expected Results
Model Type	Feature	Accuracy (Validation)
CNN	Log-Mel	~70–75%
MLP	HuBERT	~80–85%
📚 References

    CREMA-D Dataset: Kaggle

HuBERT: Hidden-Unit BERT Paper

    Course: Deep Learning for Audio and Image Applications — Dr. Rashad Hosseini, University of Tehran
