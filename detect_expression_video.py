import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm

def detect_emotions(video_path, output_path):
    print(f"Procurando vídeo em: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Erro ao abrir o vídeo.")
        return
    else:
        print("✅ Vídeo encontrado e aberto com sucesso")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("\nConfigurações do vídeo:")
    print(f"Resolução: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total de frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Erro ao criar arquivo de saída")
        cap.release()
        return
    else:
        print(f"\nArquivo de saída criado em: {output_path}")

    print("\nIniciando análise de emoções...")
    for _ in tqdm(range(total_frames), desc="Processando frames do vídeo"):
        ret, frame = cap.read()
        if not ret:
            print("\nAviso: O vídeo terminou antes do esperado")
            break

        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        for face in result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            dominant_emotion = face['dominant_emotion']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        out.write(frame)

    print("\nFinalizando processamento...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processo concluído com sucesso!\nVídeo resultante salvo em: {output_path}")

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')
output_video_path = os.path.join(script_dir, 'output_video_emotions.mp4')

detect_emotions(input_video_path, output_video_path)

 