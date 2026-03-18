"""
MoodTune — Face Emotion Detection Backend
pip install flask flask-cors deepface opencv-python-headless
python face_backend.py
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, numpy as np, cv2

app = Flask(__name__)
CORS(app)

try:
    from deepface import DeepFace
    USE_DEEPFACE = True
except ImportError:
    USE_DEEPFACE = False

EMOTION_MAP = {'happy':'happy','sad':'sad','angry':'angry','neutral':'neutral','surprise':'energetic','fear':'sad','disgust':'angry'}

def decode_image(data):
    if ',' in data: data = data.split(',')[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)

@app.route('/detect-emotion', methods=['POST'])
def detect():
    try:
        img = decode_image(request.get_json()['image'])
        if img is None: return jsonify({'error':'bad image'}), 400
        if USE_DEEPFACE:
            r = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            r = r[0] if isinstance(r, list) else r
            em = EMOTION_MAP.get(r['dominant_emotion'], 'neutral')
            return jsonify({'emotion':em, 'confidence':round(r['emotion'][r['dominant_emotion']]/100, 4)})
        else:
            import random
            ems = ['happy','sad','angry','neutral','energetic','calm']
            return jsonify({'emotion':random.choice(ems), 'confidence':round(.5+random.random()*.4,4), 'simulated':True})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status':'ok','deepface':USE_DEEPFACE})

if __name__ == '__main__':
    print(f"MoodTune Face Backend | DeepFace: {'ON' if USE_DEEPFACE else 'SIM'}")
    app.run(host='0.0.0.0', port=5000, debug=True)
