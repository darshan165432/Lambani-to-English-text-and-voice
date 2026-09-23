import numpy as np
import scipy.io.wavfile as wav

# Generate 2 seconds of random noise
sr = 16000
duration = 2
noise = np.random.uniform(-0.5, 0.5, int(sr * duration)).astype(np.float32)

wav.write("c:/major project/noise.wav", sr, noise)
print("Created noise.wav")
