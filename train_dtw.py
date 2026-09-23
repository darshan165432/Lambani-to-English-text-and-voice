import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import joblib
import os

# Define paths
METADATA_FILE = "c:/major project/metadata.csv"
TEMPLATES_FILE = "c:/major project/templates.pkl"

print("Loading metadata...")
df = pd.read_csv(METADATA_FILE)

templates = []

def trim_silence(audio, threshold=0.01):
    # Simple energy-based trim
    # Window size 100ms
    window_size = 1600 # 100ms at 16kHz
    
    if len(audio) < window_size:
        return audio
        
    energies = []
    for i in range(0, len(audio) - window_size, window_size // 2):
        chunk = audio[i:i+window_size]
        energy = np.sum(chunk**2)
        energies.append(energy)
        
    if not energies:
        return audio
        
    max_energy = np.max(energies)
    threshold_val = max_energy * threshold
    
    # Find start
    start_idx = 0
    for i, e in enumerate(energies):
        if e > threshold_val:
            start_idx = i * (window_size // 2)
            break
            
    # Find end
    end_idx = len(audio)
    for i in range(len(energies) - 1, -1, -1):
        if energies[i] > threshold_val:
            end_idx = (i + 1) * (window_size // 2) + window_size
            break
            
    trimmed = audio[start_idx:end_idx]
    if len(trimmed) < 1000: # If too short, return original or empty
        return audio
        
    return trimmed

print("Extracting feature sequences...")

for index, row in df.iterrows():
    try:
        sr, audio = wav.read(row['path'])
        
        # Convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
            
        # Trim silence
        audio = trim_silence(audio)
            
        # MFCC extraction
        mfcc_feat = mfcc(audio, sr, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512)
        
        # CMVN
        mfcc_feat = (mfcc_feat - np.mean(mfcc_feat, axis=0)) / (np.std(mfcc_feat, axis=0) + 1e-8)
        
        templates.append({
            'label': row['english'],
            'features': mfcc_feat,
            'filename': row['filename']
        })
        
    except Exception as e:
        print(f"Error processing {row['path']}: {e}")

# --- Add Garbage/Noise Templates ---
print("Generating noise templates...")
def generate_noise_features(duration=1.0, amp=0.01):
    sr = 16000
    # White noise
    noise = np.random.uniform(-amp, amp, int(sr * duration)).astype(np.float32)
    
    # MFCC
    mfcc_feat = mfcc(noise, sr, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512)
    # CMVN
    mfcc_feat = (mfcc_feat - np.mean(mfcc_feat, axis=0)) / (np.std(mfcc_feat, axis=0) + 1e-8)
    return mfcc_feat

# Add various noise types
noise_configs = [
    (0.5, 0.01), (1.0, 0.01), (2.0, 0.01), # Low noise
    (0.5, 0.05), (1.0, 0.05), (2.0, 0.05), # Medium noise
    (0.5, 0.1),  (1.0, 0.1),  (2.0, 0.1)   # High noise
]

for dur, amp in noise_configs:
    feat = generate_noise_features(dur, amp)
    templates.append({
        'label': "Not Recognized",
        'features': feat,
        'filename': f"noise_{dur}_{amp}"
    })

print(f"Total templates created: {len(templates)}")

# Save templates
joblib.dump(templates, TEMPLATES_FILE)
print(f"Templates saved to {TEMPLATES_FILE}")
