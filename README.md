# 🎧 AuthenticCall – Spoof & Fraud Call Detection

## 📌 Overview

In a nutshell, we developed a platform which can be used by **emergency helplines** to analyze the degree of **genuineness and fakeness of a call** received by them.

We analyze **vocal emotions** using **deep learning-based approaches** and deployed a web app using **Flask**.
The tool can be accessed by installing the requirements and launching `main.py`.

In this project, we are exploring **state-of-the-art models in multimodal sentiment analysis**, using text, sound, and video inputs, and developing an ensemble model that gathers information from all sources and displays it in a clear and interpretable way.

---

## 📑 Table of Contents

I. Context
II. Data Sources
III. Methodology
IV. How to use it?
V. Contributors

---

## 0. 🚀 Technologies

* **Python**
* **Flask** (Web Framework)
* **PyTorch** (Audio Classification Model)
* **Librosa** (Audio Feature Extraction)
* **NumPy & Pandas** (Data Handling)
* **Matplotlib / Plotly** (Visualization)
* **scikit-learn** (Preprocessing & Evaluation)

---

## I. 📖 Context

Affective computing is a field of Machine Learning and Computer Science that studies the recognition and processing of **human voice**.
Emotion Recognition is an old discipline, but including sound is relatively new. This field has grown with social networks providing researchers access to vast amounts of data.

---

## II. 📂 Data Sources

We are using the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**.

* **Total files:** 7356 files (~24.8 GB)
* **Actors:** 24 professional actors (12 female, 12 male)
* **Expressions:** calm, happy, sad, angry, fearful, surprise, disgust (speech), and calm, happy, sad, angry, fearful (song)
* **Intensity levels:** normal and strong, plus neutral
* **Modalities:** Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), Video-only (no sound)
* **Source:** [Zenodo RAVDESS](https://zenodo.org/record/1188976#.XCx-tc9KhQI)

---

## III. 🛠 Methodology

Our aim is to develop a model able to provide **live audio sentiment analysis** with a **visual user interface**.

### 🔹 Pipeline

1. Voice recording
2. Audio signal discretization
3. Log-mel-spectrogram extraction
4. Split spectrogram using a rolling window
5. Make a prediction using our pre-trained model

### 🔹 Model

The model is a **Time Distributed Convolutional Neural Network** (CNN).

* A rolling window is applied along the log-mel-spectrogram.
* Each window is input to a CNN with **four Local Feature Learning Blocks (LFLBs)**.
* CNN outputs are fed into a **2-cell LSTM** to learn long-term contextual dependencies.
* A fully connected layer with softmax predicts the detected emotion.

### 🔹 Overfitting Prevention

* Audio data augmentation
* Early stopping
* Keep the best model checkpoint

---

## IV. 💻 How to use it?

1. Clone the project locally
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Launch the web app:

```bash
python main.py
```

---

## V. 🤝 Contributors

* **[Anish Bhardwaj](https://github.com/21Ani)**
