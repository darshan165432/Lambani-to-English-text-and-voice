import requests
import time

url = 'http://127.0.0.1:5000/predict'
file_path = 'c:/major project/noise.wav'

print("Waiting for server...")
time.sleep(5)

try:
    with open(file_path, 'rb') as f:
        files = {'audio_data': f}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
