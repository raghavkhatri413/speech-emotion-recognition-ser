import os
import numpy as np
import streamlit as st
import librosa
import joblib
import tempfile

import tensorflow as tf
from tensorflow.keras.models import load_model

from scipy import stats

# -----------------------
#  SAME SETTINGS AS TRAINING
# -----------------------
MAX_PAD_LEN = 130

# -----------------------
#  HELPER FUNCTIONS (copied from notebook, slightly compact)
# -----------------------

def permutation_entropy(signal, order=3, delay=1):
    n = len(signal)
    if n <= order:
        return 0

    permutations = []
    for i in range(n - delay * (order - 1)):
        indices = np.argsort(signal[i:i + delay * order:delay])
        permutations.append(tuple(indices))

    if not permutations:
        return 0

    _, counts = np.unique(permutations, return_counts=True)
    probs = counts / len(permutations)
    return -np.sum(probs * np.log2(probs))


def spectral_entropy(signal, sr=22050, n_fft=2048):
    spectrogram = np.abs(librosa.stft(signal, n_fft=n_fft))
    psd = np.sum(spectrogram, axis=0)
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]
    if len(psd_norm) == 0:
        return 0
    return -np.sum(psd_norm * np.log2(psd_norm))


def renyi_entropy(signal, alpha=2):
    signal_norm = np.abs(signal) / np.sum(np.abs(signal))
    signal_norm = signal_norm[signal_norm > 0]
    if len(signal_norm) == 0:
        return 0
    if alpha == 1:
        return -np.sum(signal_norm * np.log2(signal_norm))
    return (1 / (1 - alpha)) * np.log2(np.sum(signal_norm ** alpha))


def shannon_entropy(signal):
    signal_norm = np.abs(signal) / np.sum(np.abs(signal))
    signal_norm = signal_norm[signal_norm > 0]
    if len(signal_norm) == 0:
        return 0
    return -np.sum(signal_norm * np.log2(signal_norm))


def simple_vmd_features(signal, sr=22050):
    try:
        harmonic, percussive = librosa.effects.hpss(signal)
        features = []
        for component in [signal, harmonic, percussive]:
            if len(component) > 0:
                energy = np.sum(component ** 2)
                centroid = librosa.feature.spectral_centroid(y=component, sr=sr)[0]
                bandwidth = librosa.feature.spectral_bandwidth(y=component, sr=sr)[0]
                features.extend([energy, np.mean(centroid), np.mean(bandwidth)])
            else:
                features.extend([0, 0, 0])
        return np.array(features[:9])
    except Exception:
        return np.zeros(9)


def extract_advanced_features(data, sr=22050):
    features = []
    try:
        # 1. Time-domain energy
        rms_energy = np.sqrt(np.mean(data ** 2))
        total_energy = np.sum(data ** 2)
        energy_entropy = shannon_entropy(data ** 2)
        features.extend([rms_energy, total_energy, energy_entropy])

        # 2. Spectral energy + 4 bands
        spectrogram = np.abs(librosa.stft(data, n_fft=1024))
        spectral_energy = np.sum(spectrogram ** 2)
        n_bands = 4
        band_size = spectrogram.shape[0] // n_bands
        spectral_energy_bands = []
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < n_bands - 1 else spectrogram.shape[0]
            band_energy = np.sum(spectrogram[start_idx:end_idx, :] ** 2)
            spectral_energy_bands.append(band_energy)
        features.extend([spectral_energy] + spectral_energy_bands)

        # 3. Entropies
        perm_ent = permutation_entropy(data, order=3)
        spec_ent = spectral_entropy(data, sr)
        renyi_ent = renyi_entropy(data, alpha=2)
        features.extend([perm_ent, spec_ent, renyi_ent])

        # 4. Stats + ZCR
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        zcr = np.mean(librosa.feature.zero_crossing_rate(data)[0])
        features.extend([skewness, kurtosis, zcr])

        # 5. VMD-like features
        vmd_features = simple_vmd_features(data, sr)
        features.extend(vmd_features)

        # 6. Rolloff + spectral flux
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr)[0])
        spectral_flux = np.mean(librosa.onset.onset_strength(y=data, sr=sr))
        features.extend([rolloff, spectral_flux])

        return np.array(features)

    except Exception:
        # 29 advanced features in your notebook
        return np.zeros(29)


def extract_features(file_path, max_pad_len=MAX_PAD_LEN):
    try:
        # EXACTLY like notebook: duration=3, offset=0.5, sr=22050
        data, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)

        # Traditional features
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=40, n_fft=2048, hop_length=512)
        chroma = librosa.feature.chroma_stft(y=data, sr=sr, n_fft=2048, hop_length=512)
        contrast = librosa.feature.spectral_contrast(y=data, sr=sr, n_fft=2048, hop_length=512)

        advanced_features = extract_advanced_features(data, sr)

        def pad_features(features, max_len):
            if features.shape[1] < max_len:
                pad_width = max_len - features.shape[1]
                features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                features = features[:, :max_len]
            return features

        mfcc = pad_features(mfcc, max_pad_len)
        mel_spec = pad_features(mel_spec, max_pad_len)
        chroma = pad_features(chroma, max_pad_len)
        contrast = pad_features(contrast, max_pad_len)

        combined_traditional = np.vstack([mfcc, mel_spec, chroma, contrast])

        # repeat advanced features across time steps (same as notebook)
        advanced_features_repeated = np.tile(advanced_features, (max_pad_len, 1))

        # (130, n_traditional + 29)
        combined_features = np.hstack([combined_traditional.T, advanced_features_repeated])

        return combined_features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# -----------------------
#  LOAD MODEL + ENCODERS
# -----------------------

@st.cache_resource
def load_artifacts():
    # Load full classifier model from best fold
    model = load_model("fold_4_best_model.h5", compile=False)

    # Load label encoder & scaler
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, label_encoder, scaler


model, label_encoder, scaler = load_artifacts()

# -----------------------
#  STREAMLIT UI
# -----------------------

st.title("🎙️ Speech Emotion Recognition Demo")
st.write("Upload a speech audio file (RAVDESS-style or similar) and the model will predict the emotion.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # Show the audio player
    st.audio(uploaded_file)

    # Save to a temporary file so librosa can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Analyzing audio and predicting emotion..."):
        # 1. Feature extraction
        features = extract_features(temp_path)  # (130, n_features)
        if features is None:
            st.error("Error extracting features from this audio file.")
        else:
            # 2. Scale using SAME scaler as training
            features_scaled = scaler.transform(features)      # (130, n_features)

            # 3. Add batch dimension: (1, 130, n_features)
            features_input = features_scaled[np.newaxis, ...]

            # 4. Predict
            preds = model.predict(features_input, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_emotion = label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(preds[pred_idx])

            # 5. Show result
            st.subheader("Predicted Emotion")
            st.success(f"{pred_emotion}  ({confidence*100:.2f}% confidence)")

            # Optional: show all class probabilities
            with st.expander("Show probabilities for each emotion"):
                probs = {label: float(p) for label, p in zip(label_encoder.classes_, preds)}
                st.json(probs)

    # Clean up temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass
