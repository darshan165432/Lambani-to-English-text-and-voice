import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
from python_speech_features import mfcc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Define paths
PROCESSED_DIR = "c:/major project/processed_data"
METADATA_FILE = "c:/major project/metadata.csv"
MODEL_FILE = "c:/major project/model.pkl"
ENCODER_FILE = "c:/major project/encoder.pkl"

print("Loading metadata...")
df = pd.read_csv(METADATA_FILE)

X = []
y = []

def extract_features_from_audio(audio, sr):
    try:
        # MFCC extraction
        mfcc_feat = mfcc(audio, sr, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512)
        
        # Aggregate features
        mean_feat = np.mean(mfcc_feat, axis=0)
        std_feat = np.std(mfcc_feat, axis=0)
        
        return np.concatenate((mean_feat, std_feat))
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def augment_audio(audio, sr):
    augmented_data = []
    
    # 1. Original
    augmented_data.append(audio)
    
    # 2. Speed/Pitch variations (Resampling)
    # Factors: 0.9 (Faster/Higher), 1.1 (Slower/Lower), 0.8, 1.2
    factors = [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]
    for factor in factors:
        new_len = int(len(audio) * factor)
        resampled = scipy.signal.resample(audio, new_len)
        # Pad or crop to match roughly if needed, but for MFCC it handles variable length
        augmented_data.append(resampled)
        
    # 3. Noise Injection
    # Add random noise to the original
    noise_amp = 0.005 * np.max(np.abs(audio))
    noise = np.random.normal(0, noise_amp, len(audio))
    noisy_audio = audio + noise
    augmented_data.append(noisy_audio)
    
    # 4. Volume Variation
    # Louder
    augmented_data.append(audio * 1.2)
    # Softer
    augmented_data.append(audio * 0.8)
    
    return augmented_data

print("Processing and augmenting data...")

for index, row in df.iterrows():
    try:
        sr, audio = wav.read(row['path'])
        
        # Normalize if needed (already done in prepare, but good to ensure)
        if audio.dtype != np.float32:
             if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
        
        variations = augment_audio(audio, sr)
        
        for var_audio in variations:
            feat = extract_features_from_audio(var_audio, sr)
            if feat is not None:
                X.append(feat)
                y.append(row['english'])
                
    except Exception as e:
        print(f"Error processing {row['path']}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Total training samples after augmentation: {len(X)}")
print(f"Features shape: {X.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train KNN model
# Increased neighbors to 3 since we now have multiple samples per class
print("Training improved model...")
clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
clf.fit(X, y_encoded)

# Save model and encoder
joblib.dump(clf, MODEL_FILE)
joblib.dump(le, ENCODER_FILE)

print(f"Model saved to {MODEL_FILE}")
print(f"Encoder saved to {ENCODER_FILE}")
