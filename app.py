import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import base64
import json
import traceback
import numpy as np
import cv2
import tempfile
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from deepface import DeepFace
from ultralytics import YOLO

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Custom JSON encoder: turns numpy types into plain Python types ──
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def safe_jsonify(data):
    """jsonify that handles numpy scalar types via NumpyEncoder."""
    return app.response_class(
        response=json.dumps(data, cls=NumpyEncoder),
        status=200,
        mimetype='application/json'
    )


# Initialize YOLO model globally (nano version for speed)
# Initialize YOLO model globally (Medium version for better accuracy)
yolo_model = YOLO('yolov8m.pt')

# Pre-warm DeepFace (triggers download/loading of models before server starts)
print("Pre-warming DeepFace with mediapipe (fast & accurate)...")
try:
    # Use a tiny blank image to trigger model loading
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    DeepFace.analyze(
        img_path=dummy_img,
        actions=['emotion'],
        enforce_detection=False,
        detector_backend='mediapipe',
        silent=True
    )
    print("DeepFace model (mediapipe) loaded successfully.")
except Exception as e:
    print(f"Warning: DeepFace pre-warm failed: {e}")



def preprocess_image(img):
    """
    Balanced preprocessing for both Face (DeepFace) and Objects (YOLO).
    - Uses CLAHE for contrast.
    - Uses a very mild sharpening to avoid noise.
    """
    if img is None:
        return None

    # Enhance contrast 
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) # Subtler contrast
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # No sharpening - YOLO and DeepFace often perform better on natural textures
    return img


def decode_base64_image(base64_string):
    """
    Decodes a base64 encoded image string into a numpy array (OpenCV format)
    """
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


@app.route('/')
def index():
    """Serve the main frontend HTML"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve other static files like css and js"""
    return send_from_directory(app.static_folder, path)


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    """
    API endpoint to receive a camera frame and predict emotion.
    Uses RetinaFace detector for improved face detection accuracy.
    """
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        b64_img = data['image']
        img = decode_base64_image(b64_img)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Preprocess for better accuracy
        img = preprocess_image(img)

        # --- Object Detection via YOLO ---
        phone_detected = False
        phone_boxes = []
        socializing_detected = False
        person_count = 0
        person_boxes = []
        waste_detected = False
        waste_boxes = []
        laptop_detected = False
        laptop_boxes = []

        try:
            # Lower confidence even more (0.15) to catch background people.
            yolo_results = yolo_model(img, classes=[0, 39, 41, 63, 67], conf=0.15, verbose=False)
            
            # Debug log to see detection hits
            log_entry = f"\n--- YOLO Detections ---\n"
            
            for r in yolo_results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf_score = float(box.conf[0].item())
                    b = box.xywh[0].tolist()
                    x = int(b[0] - b[2] / 2)
                    y = int(b[1] - b[3] / 2)
                    w = int(b[2])
                    h = int(b[3])
                    
                    label_map = {0: 'PERSON', 67: 'PHONE', 63: 'LAPTOP', 39: 'BOTTLE', 41: 'CUP'}
                    lbl = label_map.get(cls_id, 'UNKNOWN')
                    log_entry += f"Detected {lbl} at ({x}, {y}, {w}, {h}) with conf {conf_score:.2f}\n"

                    if cls_id == 67:  # Cell Phone
                        phone_detected = True
                        phone_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'conf': round(conf_score * 100, 1)})
                    elif cls_id == 0:  # Person
                        person_count += 1
                        person_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'conf': round(conf_score * 100, 1)})
                    elif cls_id == 63:  # Laptop
                        laptop_detected = True
                        laptop_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'conf': round(conf_score * 100, 1)})
                    elif cls_id in [39, 41]:  # Bottle or Cup
                        waste_detected = True
                        waste_boxes.append({'x': x, 'y': y, 'w': w, 'h': h,
                                            'label': 'CUP' if cls_id == 41 else 'BOTTLE',
                                            'conf': round(conf_score * 100, 1)})

            if person_count > 1:
                socializing_detected = True
                log_entry += f"SOCIALIZING FLAG SET (People count: {person_count})\n"
            
            # Write debug log (append mode)
            if log_entry.strip() != "--- YOLO Detections ---":
                with open("detection_debug.txt", "a") as df:
                    df.write(log_entry + "\n")

        except Exception as e:
            print(f"YOLO inline error: {e}")
            with open("detection_debug.txt", "a") as df:
                df.write(f"YOLO ERROR: {e}\n")

        # Use mediapipe backend
        try:
            results = DeepFace.analyze(
                img_path=img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='mediapipe',
                silent=True
            )
        except Exception as e:
            print(f"DeepFace error: {e}")
            results = []

        # Process ALL faces
        faces_data = []
        if isinstance(results, list):
            for res in results:
                reg = res.get('region', {})
                if reg.get('w', 0) > 0:
                    dom_emo = res['dominant_emotion']
                    emo_scores = {k: float(v) for k, v in res['emotion'].items()}
                    faces_data.append({
                        'emotion': str(dom_emo),
                        'confidence': round(float(res['emotion'][dom_emo]), 2),
                        'box': {
                            'x': int(reg.get('x', 0)),
                            'y': int(reg.get('y', 0)),
                            'w': int(reg.get('w', 0)),
                            'h': int(reg.get('h', 0))
                        },
                        'all_scores': emo_scores
                    })

        # Base response for backward compatibility (dominant face)
        dominant_face = faces_data[0] if faces_data else None
        
        response = {
            'success': True,
            'faces': faces_data,
            'emotion': dominant_face['emotion'] if dominant_face else '--',
            'confidence': dominant_face['confidence'] if dominant_face else 0,
            'box': dominant_face['box'] if dominant_face else {},
            'all_scores': dominant_face['all_scores'] if dominant_face else {},
            'phone_detected': bool(phone_detected),
            'phone_boxes': phone_boxes,
            'socializing_detected': bool(socializing_detected),
            'person_boxes': person_boxes,
            'waste_detected': bool(waste_detected),
            'waste_boxes': waste_boxes,
            'laptop_detected': bool(laptop_detected),
            'laptop_boxes': laptop_boxes
        }

        # If no face but objects detected, still return 200
        if not faces_data and (phone_detected or socializing_detected or waste_detected):
            return safe_jsonify(response)
        
        if not faces_data:
            return jsonify({'error': 'No face detected'}), 400

        return safe_jsonify(response)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        with open("prediction_crash.txt", "a") as f:
            f.write(f"\n--- Prediction error at {traceback.format_exc()} ---\n")
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error during prediction'}), 500


def calculate_efficiency(emotions_tally, phone_frames=0, socializing_frames=0, waste_frames=0, total_frames=0,
                          confidence_weighted=False):
    """
    Calculates an 'Employee Efficiency' score.
    Heuristic: Focused/Positive emotions (Neutral, Happy) vs Distracted/Negative.
    Penalizes for time spent on the phone, socializing, or using break items.
    """
    focused = emotions_tally.get('happy', 0) + emotions_tally.get('neutral', 0)
    total = sum(emotions_tally.values())

    if total == 0:
        return 0.0

    efficiency = (focused / total) * 100

    if total_frames > 0:
        phone_ratio = phone_frames / total_frames
        socializing_ratio = socializing_frames / total_frames
        waste_ratio = waste_frames / total_frames

        # Phone pulls penalty heavily
        phone_penalty = phone_ratio * 100
        # Socialising/Chatting penalty
        socializing_penalty = socializing_ratio * 60
        # Waste penalty
        waste_penalty = waste_ratio * 40

        efficiency -= (phone_penalty + socializing_penalty + waste_penalty)

    # Clamp to 0-100
    efficiency = max(0.0, min(100.0, efficiency))
    return round(efficiency, 2)


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """
    API endpoint to receive a video file and return aggregated emotion analysis
    and an efficiency score. Uses retinaface detector for better accuracy.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        try:
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30

            frame_interval = int(fps)  # 1 frame per second

            frame_count = 0
            analyzed_frames = 0
            phone_frames_count = 0
            socializing_frames_count = 0
            waste_frames_count = 0
            laptop_frames_count = 0

            emotions_tally = {
                'angry': 0, 'disgust': 0, 'fear': 0,
                'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0
            }

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Preprocess frame for better accuracy
                    frame = preprocess_image(frame)

                    # Analyse emotion using retinaface
                    try:
                        results = DeepFace.analyze(
                            img_path=frame,
                            actions=['emotion'],
                            enforce_detection=False,
                            detector_backend='mediapipe',
                            silent=True
                        )

                        if isinstance(results, list):
                            result = results[0]
                        else:
                            result = results

                        region = result.get('region', {})
                        if region.get('w', 0) > 0 and region.get('h', 0) > 0:
                            dom_emotion = result['dominant_emotion']
                            emotions_scores = result['emotion']
                            dom_conf = emotions_scores.get(dom_emotion, 0)

                            # Weight contribution by confidence (higher confidence = more weight)
                            weight = max(1, int(dom_conf / 20))  # 1-5 weight range
                            if dom_emotion in emotions_tally:
                                emotions_tally[dom_emotion] += weight
                            else:
                                emotions_tally[dom_emotion] = weight
                            analyzed_frames += 1

                    except Exception as e:
                        print(f"Error analyzing emotion in frame {frame_count}: {e}")

                    # Analyse objects via YOLO
                    try:
                        # 0=person, 39=bottle, 41=cup, 63=laptop, 67=phone
                        yolo_results = yolo_model(frame, classes=[0, 39, 41, 63, 67], conf=0.25, verbose=False)
                        for r in yolo_results:
                            boxes = r.boxes
                            p_cnt = 0
                            l_cnt = 0
                            w_cnt = 0
                            c_cnt = 0
                            for box in boxes:
                                cls_id = int(box.cls[0].item())
                                if cls_id == 67:
                                    c_cnt += 1
                                elif cls_id == 0:
                                    p_cnt += 1
                                elif cls_id == 63:
                                    l_cnt += 1
                                elif cls_id in [39, 41]:
                                    w_cnt += 1

                            if c_cnt > 0:
                                phone_frames_count += 1
                            if p_cnt > 1:
                                socializing_frames_count += 1
                            if w_cnt > 0:
                                waste_frames_count += 1
                            if l_cnt > 0:
                                laptop_frames_count += 1

                    except Exception as e:
                        print(f"Error analyzing objects in frame {frame_count}: {e}")

                frame_count += 1

            cap.release()

            efficiency_score = calculate_efficiency(
                emotions_tally, phone_frames_count, socializing_frames_count,
                waste_frames_count, analyzed_frames
            )

            try:
                os.remove(temp_path)
            except Exception:
                pass

            return jsonify({
                'success': True,
                'analyzed_frames': analyzed_frames,
                'phone_frames': phone_frames_count,
                'socializing_frames': socializing_frames_count,
                'waste_frames': waste_frames_count,
                'laptop_frames': laptop_frames_count,
                'emotions': emotions_tally,
                'efficiency_score': efficiency_score
            })

        except Exception as e:
            try:
                os.remove(temp_path)
            except Exception:
                pass
            print(f"Video analysis error: {e}")
            return jsonify({'error': f'Video processing error: {str(e)}'}), 500

    return jsonify({'error': 'Unknown error saving file'}), 500


if __name__ == '__main__':
    os.makedirs(app.static_folder, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
