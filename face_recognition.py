import cv2
import numpy as np
import os
from scipy.spatial.distance import cosine
from deepface import DeepFace
from tqdm import tqdm

class EnhancedFaceRecognition:
    def __init__(self):
        """Initialize facial recognition models and settings."""
        self.initialize_models()
        self.confidence_threshold = 0.7  # Threshold for face detection confidence
        self.face_size = 96  # Standard size for face alignment
        
    def initialize_models(self):
        """Load and initialize the required models."""
        print("Loading face detection and recognition models...")
        # Load face detection model
        prototxt_path = "deploy.prototxt"
        caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.face_detector = cv2.dnn.readNet(prototxt_path, caffemodel_path)
        
        # Load face recognition model
        self.face_recognizer = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")
        
    def preprocess_frame(self, frame):
        """Preprocess frame for face detection."""
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            [104, 117, 123], 
            swapRB=False, 
            crop=False
        )
        return blob
        
    def detect_and_align_face(self, frame):
        """Detect and align faces in the frame."""
        height, width = frame.shape[:2]
        blob = self.preprocess_frame(frame)
        
        # Detect faces
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        aligned_faces = []
        face_locations = []
        
        # Process each detection
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Get face coordinates
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype("int")
                
                # Extract and align face
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    aligned_face = cv2.resize(face_roi, (self.face_size, self.face_size))
                    aligned_faces.append(aligned_face)
                    face_locations.append((x1, y1, x2, y2))
                    
        return aligned_faces, face_locations

    def get_face_embedding(self, face):
        """Extract facial embedding using the recognition model."""
        # Prepare face for recognition
        face_blob = cv2.dnn.blobFromImage(
            face, 
            1.0/255, 
            (self.face_size, self.face_size),
            (0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        
        # Get embedding
        self.face_recognizer.setInput(face_blob)
        embedding = self.face_recognizer.forward()
        return embedding.flatten()

    def analyze_emotions(self, frame, face_location):
        """Analyze emotions in detected face using DeepFace."""
        try:
            # Extract face ROI
            x1, y1, x2, y2 = face_location
            face_roi = frame[y1:y2, x1:x2]
            
            # Analyze emotions
            analysis = DeepFace.analyze(
                face_roi, 
                actions=['emotion'],
                enforce_detection=False
            )
            
            return analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            return None

    def process_video(self, input_path, output_path, known_face_encodings=None, known_face_names=None):
        """Process video for face recognition and emotion detection."""
        print("Processing video...")
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Detect and align faces
                aligned_faces, face_locations = self.detect_and_align_face(frame)
                
                if not aligned_faces:
                    # Add message when no faces are detected
                    cv2.putText(frame, "No faces detected", (20, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # Process each detected face
                for face, location in zip(aligned_faces, face_locations):
                    x1, y1, x2, y2 = location
                    
                    # Get emotion
                    emotion = self.analyze_emotions(frame, location)
                    
                    # Get face embedding for recognition
                    face_embedding = self.get_face_embedding(face)
                    
                    # Initialize name as "Unknown"
                    name = "Unknown"
                    
                    # Compare with known faces if provided
                    if known_face_encodings is not None and known_face_names is not None:
                        # Find the best match
                        face_distances = [cosine(face_embedding, known_enc) for known_enc in known_face_encodings]
                        best_match_index = np.argmin(face_distances)
                        
                        if face_distances[best_match_index] < 0.3:  # Threshold for recognition
                            name = known_face_names[best_match_index]
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add name and emotion labels
                    label = f"{name} | Emotion: {emotion}" if emotion else name
                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                              
                    # If face is unknown, add notification
                    if name == "Unknown":
                        cv2.putText(frame, "No known face identified", (20, height - 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # Write frame
                out.write(frame)
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        print("Video processing completed!")

def download_models():
    """Download required pre-trained models."""
    import urllib.request
    
    models = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "openface.nn4.small2.v1.t7": "http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.t7"
    }
    
    print("Downloading required models...")
    for filename, url in models.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"{filename} downloaded successfully!")

def load_known_faces(images_folder):
    """Load and encode known faces from images folder.
    
    Args:
        images_folder (str): Path to folder containing known face images
        
    Returns:
        tuple: Lists of face encodings and corresponding names
    """
    known_face_encodings = []
    known_face_names = []
    
    print("\nLoading known faces from images folder...")
    
    try:
        # Verify folder exists
        if not os.path.exists(images_folder):
            print(f"Warning: Images folder '{images_folder}' not found!")
            return [], []
            
        # Process each image in folder
        for filename in os.listdir(images_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_folder, filename)
                
                # Extract name from filename (remove extension and numbers)
                name = os.path.splitext(filename)[0]
                name = ''.join([i for i in name if not i.isdigit()])
                
                # Read and process image
                print(f"Processing {filename}...")
                image = cv2.imread(image_path)
                faces, locations = EnhancedFaceRecognition().detect_and_align_face(image)
                
                if faces:
                    # Get encoding of first face found
                    face_encoding = EnhancedFaceRecognition().get_face_embedding(faces[0])
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
    """Main execution function."""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup paths
    input_video = os.path.join(script_dir, 'video.mp4')
    output_video = os.path.join(script_dir, 'output_video_enhanced.mp4')
    images_folder = os.path.join(script_dir, 'images')
    
    try:
        # Download required models
        download_models()
        
        # Load known faces
        known_face_encodings, known_face_names = load_known_faces(images_folder)
        
        # Initialize and run face recognition
        face_recognizer = EnhancedFaceRecognition()
        face_recognizer.process_video(
            input_video, 
            output_video,
            known_face_encodings,
            known_face_names
        )
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main()