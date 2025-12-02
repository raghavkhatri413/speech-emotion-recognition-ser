---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "4.42.0"
app_file: app.py
pinned: false
---

# Speech Emotion Recognition (RAVDESS)

Upload a short speech audio clip and this Space will predict the **emotion** using a
hybrid Autoencoder–CNN–LSTM model trained on the **RAVDESS** dataset.

**How to use:**

1. Click on **Upload** and select a `.wav` (or compatible) audio file containing speech (around 3 seconds works best).
2. Click **Submit / Predict**.
3. The app will return:
   - Predicted emotion label
   - Confidence score
   - Class-wise probabilities
