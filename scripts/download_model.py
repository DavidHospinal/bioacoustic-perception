
import os
import urllib.request
import ssl

# Bypass SSL verification if needed
ssl._create_default_https_context = ssl._create_unverified_context

# YAMNet TFLite model (direct storage link)
MODEL_URL = "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/yamnet/classification/tflite/1.tflite"

# Class map CSV
CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "yamnet.tflite")
CLASS_MAP_PATH = os.path.join(MODELS_DIR, "yamnet_class_map.csv")

def download_file(url, path):
    print(f"Downloading {url} to {path}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"Successfully downloaded to {path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    if not os.path.exists(MODELS_DIR):
        print(f"Creating directory: {MODELS_DIR}")
        os.makedirs(MODELS_DIR)

    if not os.path.exists(MODEL_PATH):
        download_file(MODEL_URL, MODEL_PATH)
    else:
        print(f"Model already exists at {MODEL_PATH}")

    if not os.path.exists(CLASS_MAP_PATH):
        download_file(CLASS_MAP_URL, CLASS_MAP_PATH)
    else:
        print(f"Class map already exists at {CLASS_MAP_PATH}")

if __name__ == "__main__":
    main()
