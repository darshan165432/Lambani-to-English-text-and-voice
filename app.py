from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import joblib
from moviepy import AudioFileClip
import uuid
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import webbrowser
from threading import Timer

app = Flask(__name__)

# Load templates
TEMPLATES_FILE = "c:/major project/templates.pkl"
templates = joblib.load(TEMPLATES_FILE)

UPLOAD_FOLDER = "c:/major project/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
    if len(trimmed) < 1000: 
        return audio
        
    return trimmed

def extract_features_sequence(file_path):
    try:
        # Load and resample
        temp_wav = file_path + "_temp.wav"
        
        try:
            clip = AudioFileClip(file_path)
            clip.write_audiofile(temp_wav, codec='pcm_s16le', fps=16000)
            clip.close()
            sr, audio = wav.read(temp_wav)
        except Exception as e:
            print(f"Moviepy failed, trying direct read: {e}")
            sr, audio = wav.read(file_path)

        # Clean up temp
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        # Convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Normalize audio
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
            
        # Check duration (ignore < 0.5s)
        duration = len(audio) / sr
        print(f"Duration: {duration}s")
        if duration < 0.5:
            print("Audio too short.")
            return None

        # Check for absolute silence/low noise
        max_amp = np.max(np.abs(audio))
        print(f"Max Amplitude: {max_amp}")
        
        if max_amp < 0.1: # Increased to 10%
            print("Audio too quiet, treated as silence.")
            return None
            
        # Trim silence
        audio = trim_silence(audio)
            
        # MFCC extraction
        mfcc_feat = mfcc(audio, sr, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512)
        
        # CMVN
        mfcc_feat = (mfcc_feat - np.mean(mfcc_feat, axis=0)) / (np.std(mfcc_feat, axis=0) + 1e-8)
        
        return mfcc_feat
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio_data']
    filename = str(uuid.uuid4()) + ".webm"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Extract features
        input_features = extract_features_sequence(filepath)
        
        if input_features is None:
            return jsonify({'translation': "No speech detected"})
            
        # Compare with templates using DTW
        best_dist = float('inf')
        best_label = "Unknown"
        all_results = []
        
        for template in templates:
            ref_features = template['features']
            
            # Optimization: Skip if length difference is too big (>50%)
            # This avoids matching short words to long sentences
            len_diff = abs(len(input_features) - len(ref_features))
            if len_diff > 0.5 * max(len(input_features), len(ref_features)):
                continue
                
            distance, path = fastdtw(input_features, ref_features, dist=euclidean)
            
            # Normalize distance by length of path to be fair to longer words
            normalized_dist = distance / len(path)
            
            # Store all results
            all_results.append({'label': template['label'], 'dist': normalized_dist})
            
            if normalized_dist < best_dist:
                best_dist = normalized_dist
                best_label = template['label']
        
        # Sort and print top 5
        all_results.sort(key=lambda x: x['dist'])
        print("Top 5 matches:")
        for res in all_results[:5]:
            print(f"{res['label']}: {res['dist']}")

        # Threshold for rejection
        
        # Threshold for rejection
        # Threshold for rejection
        # Strict threshold now that we have garbage classes
        THRESHOLD = 3.5
        
        if best_dist > THRESHOLD or best_label == "Not Recognized":
            return jsonify({'translation': "Not Recognized"})
            
        return jsonify({'translation': best_label})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, port=5000)
