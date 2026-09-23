import os
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
from moviepy import VideoFileClip, AudioFileClip
import shutil

# Define paths
DATASET_DIR = "c:/major project/lamani_translation_dataset"
PROCESSED_DIR = "c:/major project/processed_data"
METADATA_FILE = "c:/major project/metadata.csv"
TARGET_SR = 16000

# Create processed directory
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

data = []

print(f"Processing files from {DATASET_DIR}...")

def load_and_resample(path, target_sr=16000):
    try:
        sr, audio = wav.read(path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Normalize to float -1 to 1
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0
            
        # Resample
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = scipy.signal.resample(audio, num_samples)
            
        return audio, target_sr
    except Exception as e:
        print(f"Error reading wav {path}: {e}")
        return None, None

for filename in os.listdir(DATASET_DIR):
    file_path = os.path.join(DATASET_DIR, filename)
    
    # Skip directories
    if os.path.isdir(file_path):
        continue
        
    # Parse filename for labels
    name_part = os.path.splitext(filename)[0]
    if "," in name_part:
        parts = name_part.split(",")
        lamani_text = parts[0].strip()
        english_text = parts[1].strip()
    else:
        lamani_text = name_part
        english_text = name_part

    target_filename = f"{lamani_text}_{english_text}.wav".replace(" ", "_")
    target_path = os.path.join(PROCESSED_DIR, target_filename)
    
    try:
        temp_wav = "temp_convert.wav"
        
        # Convert to WAV first
        if filename.lower().endswith(".mp4"):
            clip = AudioFileClip(file_path)
            clip.write_audiofile(temp_wav, codec='pcm_s16le', fps=TARGET_SR)
            clip.close()
            # It's already resampled by moviepy if fps is set
            sr, audio = wav.read(temp_wav)
            # Normalize
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            os.remove(temp_wav)
        elif filename.lower().endswith(".ogg"):
             # Moviepy can handle ogg too
            clip = AudioFileClip(file_path)
            clip.write_audiofile(temp_wav, codec='pcm_s16le', fps=TARGET_SR)
            clip.close()
            sr, audio = wav.read(temp_wav)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            os.remove(temp_wav)
        elif filename.lower().endswith(".wav"):
            audio, sr = load_and_resample(file_path, TARGET_SR)
            if audio is None: continue
        else:
            print(f"Skipping unknown format: {filename}")
            continue

        # Save processed wav
        # Convert back to int16 for saving
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(target_path, TARGET_SR, audio_int16)
            
        data.append({
            "filename": target_filename,
            "lamani": lamani_text,
            "english": english_text,
            "path": target_path
        })
        print(f"Processed: {filename}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        if os.path.exists("temp_convert.wav"):
            os.remove("temp_convert.wav")

# Save metadata
df = pd.DataFrame(data)
df.to_csv(METADATA_FILE, index=False)
print(f"Metadata saved to {METADATA_FILE}")
print(f"Total files processed: {len(df)}")
