# AuthentiCall Detection using Emotion Analysis

## 📌 Overview
We developed a platform for **emergency helplines** to analyze the **degree of genuineness or fakeness** of a call they receive.  
The system analyzes **vocal emotions** using **deep learning** and is deployed as a **Flask web application**.

The tool can be accessed by installing the requirements and launching `main.py`.

---

## 📑 Table of Contents
1. [Context](#context)
2. [Data Sources](#data-sources)
3. [Methodology](#methodology)
4. [How to Use](#how-to-use)
5. [Contributors](#contributors)
6. [Technologies](#technologies)

---

## 0. 🚀 Technologies
- **Python**
- **Flask**
- **Deep Learning** (Time Distributed CNN + LSTM)
- **Librosa** (Audio Processing)
- **RAVDESS Dataset**
- **TESS Dataset**

---

## I. 📖 Context
Affective computing is a field of Machine Learning and Computer Science that studies the recognition and processing of **human voice**.  

While **emotion recognition** has existed for years, **sound-based** analysis is relatively new.  
The rise of social networks has provided researchers with **massive datasets** to train these systems.

---

## II. 📂 Data Sources

### 1️⃣ Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- **Total Files:** 7,356  
- **Size:** 24.8 GB  
- **Actors:** 24 professional actors (12 female, 12 male)  
- **Emotions (Speech):** calm, happy, sad, angry, fearful, surprise, disgust, neutral  
- **Emotions (Song):** calm, happy, sad, angry, fearful  
- **Formats:** Audio-only (.wav), Audio-Video (.mp4), Video-only (no sound)  
- **Source:** [RAVDESS on Zenodo](https://zenodo.org/record/1188976#.XCx-tc9KhQI)

---

### 2️⃣ Toronto Emotional Speech Set (TESS)
- **Total Files:** 2,800+  
- **Actors:** 2 female actors aged 26 and 64  
- **Emotions:** anger, disgust, fear, happiness, pleasant surprise, sadness, neutral  
- **Format:** Audio-only (.wav)  
- **Source:** [TESS Dataset on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

---

## III. 🛠 Methodology
Our goal was to **provide live audio sentiment analysis** with a clear visual interface.

### 🔹 Pipeline
1. Voice Recording  
2. Audio Signal Discretization  
3. **Log-Mel-Spectrogram Extraction**  
4. Split Spectrogram using Rolling Window  
5. Predict Emotions using Pre-trained Model  

### 🔹 Model Architecture
We used a **Time Distributed Convolutional Neural Network (CNN)** with the following:
- **Local Feature Learning Blocks (LFLBs)** for feature extraction  
- **LSTM Layers** for long-term contextual dependencies  
- **Softmax Layer** for final emotion classification  

**Overfitting Prevention:**
- Audio Data Augmentation  
- Early Stopping  
- Best Model Checkpointing  

---

## IV. 💻 How to Use
```bash
# Clone the repository
git clone <your-repo-url>
cd <your-project-folder>

# Install dependencies
pip install -r requirements.txt

# Run the web app
python main.py
