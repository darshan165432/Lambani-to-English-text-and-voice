import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
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

print("Extracting features...")

def extract_features(file_path):
    try:
        sr, audio = wav.read(file_path)
        # MFCC extraction
        # winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512
        mfcc_feat = mfcc(audio, sr, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512)
        
        # Aggregate features (Mean and Std) to handle variable length
        mean_feat = np.mean(mfcc_feat, axis=0)
        std_feat = np.std(mfcc_feat, axis=0)
        
        # Concatenate mean and std
        feature_vector = np.concatenate((mean_feat, std_feat))
        return feature_vector
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

for index, row in df.iterrows():
    feat = extract_features(row['path'])
    if feat is not None:
        X.append(feat)
        y.append(row['english'])

X = np.array(X)
y = np.array(y)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train KNN model (k=1 because we have 1 sample per class)
print("Training model...")
clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
clf.fit(X, y_encoded)

# Save model and encoder
joblib.dump(clf, MODEL_FILE)
joblib.dump(le, ENCODER_FILE)

print(f"Model saved to {MODEL_FILE}")
print(f"Encoder saved to {ENCODER_FILE}")
