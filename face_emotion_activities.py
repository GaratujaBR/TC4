import cv2
import numpy as np
import os
import torch
from scipy.spatial.distance import cosine
from deepface import DeepFace
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import urllib.request

class CombinedVideoAnalyzer:
    def __init__(self):
        """Initialize all models and settings."""
        self.initialize_face_models()
        self.initialize_activity_models()
        self.confidence_threshold = 0.7
        self.face_size = 96
        self.prediction_buffer = []
        self.buffer_size = 5
        
    def initialize_face_models(self):
        """Load face detection and recognition models."""
        print("Loading face detection and recognition models...")
        prototxt_path = "deploy.prototxt"
        caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.face_detector = cv2.dnn.readNet(prototxt_path, caffemodel_path)
        self.face_recognizer = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")
        
    def initialize_activity_models(self):
        """Load activity recognition models."""
        print("Loading activity recognition models...")
        self.model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activity_model = ViTForImageClassification.from_pretrained(self.model_name).to(self.device)
        self.activity_processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.activity_model.eval()

    def detect_and_align_face(self, frame):
        """Detect and align faces in the frame."""
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            [104, 117, 123], 
            swapRB=False, 
            crop=False
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        aligned_faces = []
        face_locations = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype("int")
                
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    aligned_face = cv2.resize(face_roi, (self.face_size, self.face_size))
                    aligned_faces.append(aligned_face)
                    face_locations.append((x1, y1, x2, y2))
                    
        return aligned_faces, face_locations

    def smooth_predictions(self, prediction, confidence):
        """Aplica suavização temporal às predições."""
        self.prediction_buffer.append((prediction, confidence))
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
            
        # Contar ocorrências de cada predição
        prediction_counts = {}
        total_confidence = {}
        
        for pred, conf in self.prediction_buffer:
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                total_confidence[pred] = 0
            prediction_counts[pred] += 1
            total_confidence[pred] += conf
            
        # Encontrar a predição mais frequente
        max_count = 0
        smoothed_prediction = prediction
        smoothed_confidence = confidence
        
        for pred, count in prediction_counts.items():
            if count > max_count:
                max_count = count
                smoothed_prediction = pred
                smoothed_confidence = total_confidence[pred] / count
                
        return smoothed_prediction, smoothed_confidence

    def get_face_embedding(self, face):
        """Extract facial embedding using the recognition model."""
        face_blob = cv2.dnn.blobFromImage(
            face, 
            1.0/255, 
            (self.face_size, self.face_size),
            (0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        
        self.face_recognizer.setInput(face_blob)
        embedding = self.face_recognizer.forward()
        return embedding.flatten()

    def analyze_emotions(self, frame, face_location):
        """Analyze emotions in detected face using DeepFace."""
        try:
            x1, y1, x2, y2 = face_location
            face_roi = frame[y1:y2, x1:x2]
            
            analysis = DeepFace.analyze(
                face_roi, 
                actions=['emotion'],
                enforce_detection=False
            )
            
            return analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            return None

    def process_activity(self, frame):
        """Process frame for activity recognition."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.activity_processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.activity_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        confidence, prediction = torch.max(probs, 1)
        activity = self.activity_model.config.id2label[prediction.item()]
        
        # Dicionário de tradução de atividades
        activity_translations = {
            'dancing': 'dancando',
            'sleeping': 'dormindo',
            'clapping': 'aplaudindo',
            'drinking': 'bebendo',
            'laughing': 'rindo',
            'eating': 'comendo',
            'sitting': 'sentado',
            'standing': 'em pé',
            'walking': 'andando',
            'running': 'correndo',
            'jumping': 'pulando',
            'fighting': 'brigando',
            'climbing': 'escalando',
            'reading': 'lendo',
            'writing': 'escrevendo',
            'cooking': 'cozinhando',
            'talking': 'falando',
            'texting': 'mexendo no celular',
            'playing': 'brincando',
            'working': 'trabalhando',
            'exercising': 'se exercitando',
            'cleaning': 'limpando',
            'cycling': 'pedalando',
            'driving': 'dirigindo',
            'shopping': 'comprando',
            'singing': 'cantando',
            'hugging': 'abraçando',
            'kissing': 'beijando',
            'pushing': 'empurrando',
            'pulling': 'puxando',
            'carrying': 'carregando',
            'throwing': 'arremessando',
            'catching': 'pegando',
            'lifting': 'levantando peso'
        }
        
        # Traduzir a atividade ou manter original se não houver tradução
        translated_activity = activity_translations.get(activity.lower(), activity)
        
        return translated_activity, confidence.item()

    def process_video(self, input_path, output_path, known_face_encodings=None, known_face_names=None):
        """Process video for face recognition, emotion detection, and activity recognition."""
        print("Processing video...")
        cap = cv2.VideoCapture(input_path)
        
        # Video writer setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        current_action = "Unknown"
        current_confidence = 0.0
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Activity Recognition
                action, confidence = self.process_activity(frame)
                action, confidence = self.smooth_predictions(action, confidence)
                if confidence > 0.5:
                    current_action = action
                    current_confidence = confidence
                
                # Face Detection and Emotion Recognition
                aligned_faces, face_locations = self.detect_and_align_face(frame)
                
                if not aligned_faces:
                    cv2.putText(frame, "Nenhum rosto detectado", (20, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # Process detected faces
                for face, location in zip(aligned_faces, face_locations):
                    x1, y1, x2, y2 = location
                    emotion = self.analyze_emotions(frame, location)
                    face_embedding = self.get_face_embedding(face)
                    name = "Nao Identificado"
                    
                    if known_face_encodings is not None and known_face_names is not None:
                        face_distances = [cosine(face_embedding, known_enc) for known_enc in known_face_encodings]
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < 0.3:
                            name = known_face_names[best_match_index]
                    
                    # Draw results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    emotion_translations = {
                        'happy': 'feliz', 'sad': 'triste', 'angry': 'irritado',
                        'fear': 'medo', 'surprise': 'surpreso', 'neutral': 'neutro',
                        'disgust': 'nojo'
                    }
                    
                    if emotion:
                        emotion = emotion_translations.get(emotion.lower(), 'anomalia')
                    
                    label = f"{name} | Emocao: {emotion}" if emotion else name
                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                # Draw activity recognition results
                self.draw_activity_results(frame, current_action, current_confidence)
                
                out.write(frame)
                pbar.update(1)
        
        cap.release()
        out.release()
        print("Video processing completed!")

    def draw_activity_results(self, frame, action, confidence):
        """Draw activity recognition results on frame."""
        height = frame.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (int(255 * (1 - confidence)), int(255 * confidence), 0)
        
        cv2.putText(frame, f"Atividade: {action}", (20, height - 60),
                   font, 1, color, 2)
        cv2.putText(frame, f"Confianca: {confidence:.2f}", (20, height - 20),
                   font, 1, color, 2)

def download_models():
    """Download required pre-trained models."""
    models = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "openface.nn4.small2.v1.t7": "http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.t7"
    }
    
    print("Downloading required models...")
    for filename, url in models.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"{filename} downloaded successfully!")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                raise

def load_known_faces(images_folder):
    """Load and encode known faces from images folder."""
    known_face_encodings = []
    known_face_names = []
    
    print("\nLoading known faces from images folder...")
    
    try:
        if not os.path.exists(images_folder):
            print(f"Warning: Images folder '{images_folder}' not found!")
            return [], []
            
        for filename in os.listdir(images_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_folder, filename)
                
                name = os.path.splitext(filename)[0]
                name = ''.join([i for i in name if not i.isdigit()])
                
                print(f"Processing {filename}...")
                image = cv2.imread(image_path)
                
                recognizer = CombinedVideoAnalyzer()
                faces, locations = recognizer.detect_and_align_face(image)
                
                if faces:
                    face_encoding = recognizer.get_face_embedding(faces[0])
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                    print(f"Successfully encoded face for {name}")
                else:
                    print(f"No face detected in {filename}")
                    
        print(f"\nLoaded {len(known_face_encodings)} known faces")
        return known_face_encodings, known_face_names
        
    except Exception as e:
        print(f"Error loading known faces: {str(e)}")
        return [], []

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video = os.path.join(script_dir, 'video.mp4')
    output_video = os.path.join(script_dir, 'output_combined.mp4')
    images_folder = os.path.join(script_dir, 'images')
    
    try:
        download_models()
        known_face_encodings, known_face_names = load_known_faces(images_folder)
        
        analyzer = CombinedVideoAnalyzer()
        analyzer.process_video(
            input_video,
            output_video,
            known_face_encodings,
            known_face_names
        )
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main()